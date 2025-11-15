import torch
import random
import torch.nn as nn
import numpy as np
import os
import quantize.int_linear_fake as int_linear_fake
import quantize.int_linear_real as int_linear_real
from quantize.recon_loss import get_recon_loss

from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import gc
from utils.quant_utils import (
    clip_parameters,weight_lora_parameters,trainable_parameters,set_clip_parameters,set_weight_lora_parameters,
    set_quant_state,quant_inplace, trainable_parameters_num,get_named_linears,set_op_by_name, model_bit_refactor,
    set_grad_mask,remove_grad_mask)
import time
from utils import train_utils
from utils.train_utils import NativeScalerWithGradNormCount
from utils.data_utils import BlockTrainDataset, copy_block_dataset
from contextlib import nullcontext
from utils.model_utils import get_kv_cache, mv_kv_cache

from awq.quantize.pre_quant import get_blocks, move_embed, process_input
import copy

def cosine_similarity(input1:torch.Tensor, input2:torch.Tensor, top_k = 0):
    normalized_input1 = input1 / input1.norm(dim=-1, keepdim=True)
    normalized_input2 = input2 / input2.norm(dim=-1, keepdim=True)
    similarity = (normalized_input1 * normalized_input2).sum(dim=-1)
    if top_k > 0:
        _, idx = torch.topk(similarity, k=top_k)
        return similarity, idx
    else:
        return similarity
    
def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()

def save_output_by_layer(layer_id, out, save_path):
    folder_name = f"layer_{layer_id}"
    
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    
    torch.save(out, save_path + "q_out_" + '{}.pt'.format(folder_name))

@torch.no_grad()
def update_dataset(layer, dataset, dev, layer_kwargs, is_token_fusion = False, 
                   quant_type = 'weight_only',topk_token = 0.5):
    with torch.no_grad(): 
        with torch.cuda.amp.autocast():
            for index, inps in enumerate(dataset):   
                inps = inps.to(dev)
                layer_kwarg = layer_kwargs[index]
                if len(inps.shape)==2:
                    inps = inps.unsqueeze(0)
                batch_feature = []
                for i in range(inps.size(0)):#
                    _ = []
                    if quant_type == 'weight_only': # for weight only use 2-8, 4 groups
                        bit_groups = [[2],[3,4],[5,6],[7,8]]
                        for group_index, bit_choices in zip(range(len(bit_groups)), bit_groups):
                            b = random.choice(bit_choices)
                            model_bit_refactor(layer, b)
                            _.append(layer(inps[i:i+1], **layer_kwarg)[0])
                    else:
                        bit_groups = [[4],[5,6,],[7,8]] # ,[5,6,],[7,8]
                        for group_index, bit_choices in zip(range(len(bit_groups)), bit_groups):
                            b = random.choice(bit_choices)
                            model_bit_refactor(layer, b)
                            _.append(layer(inps[i:i+1], **layer_kwarg)[0])
                    B, N, C = _[0].shape
                    
                    for index_bit in range(len(_)):
                        _[index_bit] = _[index_bit].reshape(-1, _[index_bit].size(-1))
                    
                    f = []
                    k = int((topk_token) * N)
                    similarity, top_k_token = cosine_similarity(_[0], _[-1], top_k=k)
                    
                    for t in range(N):
                        if t in top_k_token.tolist():
                            f.append(_[-1][t])
                        else:
                            if quant_type == 'weight_only':
                                mfm_lambda = 0.25
                                if torch.isnan((_[0][t] + _[1][t] + _[2][t] + _[3][t]) * mfm_lambda).any():
                                    import pdb 
                                    pdb.set_trace() 
                                f.append((_[0][t] + _[1][t] + _[2][t] + _[3][t]) * mfm_lambda)
                            else:
                                mfm_lambda = [0.4,0.3,0.3]
                                f.append(_[0][t] * mfm_lambda[0] + _[1][t] * mfm_lambda[1]+ _[2][t] * mfm_lambda[2])
            
                    f = torch.cat([x for x in f])
                    f = f.reshape(B, N, C)
                    batch_feature.append(f)
                    
                new_data = torch.cat([x for x in batch_feature]).to('cpu')
                dataset[index] = new_data
    torch.cuda.empty_cache()
    train_utils.cleanup_memory()
            
def train_one_epoch(qlayer, layer_kwargs,
                      loss_scaler, loss_func, lr_schedule, optimizer, dev, traincast,
                      quant_inps, fp_inps_with_fp, block_index, args):
    loss_list = []
    norm_list = []
    for index in range(len(quant_inps)): # batch-wise
        with traincast():
            input = quant_inps[index].to(dev)
            label = fp_inps_with_fp[index].to(dev)
            layer_kwarg = layer_kwargs[index]
            if len(input.shape)==2:
                    input = input.unsqueeze(0)
            loss = None
            cached_out = None
            if args.quant_type == 'weight_only': # for weight only use 2-8, 4 groups
                bit_groups = [[7,8],[5,6],[3,4],[2]] 
                for group_index, bit_choices in zip(range(len(bit_groups)), bit_groups):
                    bit = random.choice(bit_choices)
                    model_bit_refactor(qlayer, bit)
                    quant_out = qlayer(input, **layer_kwarg)[0]
                    if group_index == 0:# high group
                        loss = lp_loss(quant_out, label, p=1)
                        
                    else:
                        loss += lp_loss(quant_out, label, p=1)
                    # if not math.isfinite(loss.item()):
                    #     import pdb;pdb.set_trace()
            elif args.quant_type == 'weight_act':       
                bit_groups = [[7,8],[5,6,],[4]] ## bit_groups = [[7,8],[5,6,],[4]] #
                for group_index, bit_choices in zip(range(len(bit_groups)), bit_groups):
                    bit = random.choice(bit_choices)
                    model_bit_refactor(qlayer, bit)
                    quant_out = qlayer(input, **layer_kwarg)[0]
                    if group_index == 0:# high group
                        loss = lp_loss(quant_out, label, p=1) # mse loss
                        if args.token_dist_loss:
                            cached_out = quant_out # cache high bit out
                    else:
                        loss += lp_loss(quant_out, label, p=1) # mse loss
                    # if not math.isfinite(loss.item()):
                    #     import pdb;pdb.set_trace()      

        if not math.isfinite(loss.item()):
            print("Loss is NAN, stopping training")
        loss_list.append(loss.detach().cpu())
        optimizer.zero_grad()
        norm = loss_scaler(loss, optimizer,parameters=trainable_parameters(qlayer, args)).cpu()
        norm_list.append(norm.data)
        lr_schedule.step(optimizer)
    loss_mean = torch.stack(loss_list).mean()
    norm_mean = torch.stack(norm_list).mean()
    return loss_mean, norm_mean

@torch.no_grad()
def eval_one_epoch(qlayer, layer_kwargs,
                      loss_func, dev, traincast,
                      quant_inps, fp_inps_with_fp, args):
    loss_list_low = []
    loss_list_high = []
    for index in range(len(quant_inps)):
        with traincast():
            layer_kwarg = layer_kwargs[index]
            input = quant_inps[index].to(dev)
            label = fp_inps_with_fp[index].to(dev)
            if args.quant_type == 'weight_only': # for weight only use 2-8, 4 groups
                bit_groups = [[2],[7,8]]
                for group_index, bit_choices in zip(range(len(bit_groups)), bit_groups):
                    bit = random.choice(bit_choices)
                    model_bit_refactor(qlayer, bit)
                    quant_out = qlayer(input, **layer_kwarg)[0]
                    if group_index == 0:
                        loss = loss_func(quant_out, label)
                        loss_list_low.append(loss.detach().cpu())
                    else:
                        loss = loss_func(quant_out, label)
                        loss_list_high.append(loss.detach().cpu())
            elif args.quant_type == 'weight_act':       
                bit_groups = [[4],[7,8]]
                for group_index, bit_choices in zip(range(len(bit_groups)), bit_groups):
                    bit = random.choice(bit_choices)
                    model_bit_refactor(qlayer, bit)
                    quant_out = qlayer(input, **layer_kwarg)[0]
                    if group_index == 0:
                        loss = loss_func(quant_out, label)
                        loss_list_low.append(loss.detach().cpu())
                    else:
                        loss = loss_func(quant_out, label)
                        loss_list_high.append(loss.detach().cpu())
        
    loss_low_mean = torch.stack(loss_list_low).mean()
    loss_high_mean = torch.stack(loss_list_high).mean()
    return loss_low_mean, loss_high_mean

class CustomLRSchedule(object):
    def __init__(self, args, total_iter) -> None:
        param_group_index = 0
        if args.clip_lr > 0:
            empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=args.clip_lr)
            self.quant_scheduler = CosineAnnealingLR(empty_optimizer_1, T_max=total_iter, eta_min=args.clip_lr/args.min_lr_factor)#
            self.quant_index = param_group_index
            param_group_index += 1
        else:
            self.quant_scheduler = None
        if args.weight_lr > 0:
            empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=args.weight_lr)
            self.weight_scheduler = CosineAnnealingLR(empty_optimizer_2, T_max=total_iter, eta_min=args.weight_lr/args.min_lr_factor)
            self.weight_index = param_group_index
            param_group_index += 1  
        else:
            self.weight_scheduler = None
    def step(self, optimizer):
        if self.quant_scheduler is not None:
            self.quant_scheduler.step()
            optimizer.param_groups[self.quant_index]['lr'] = self.quant_scheduler.get_lr()[0]
        if self.weight_scheduler is not None:
            self.weight_scheduler.step()
            optimizer.param_groups[self.weight_index]['lr'] = self.weight_scheduler.get_lr()[0]
        
             
def block_ap(
    model,
    args,
    prompt_inputs,
    prompt_kwargs,
    logger=None,
):
    logger.info("Starting ...")
    logger.info(f"Lora Type -- {args.lora_type}")
    
    dev = args.device
    # use_cache = model.config.use_cache
    # model.config.use_cache = False
    model_dtype = next(model.model.parameters()).dtype
    layer_kwargs_list = []
    layer_kwargs = {}
    # step 1: move embedding layer and first layer to target device
    layers = get_blocks(model.model)
    move_embed(model.model, dev)

    layers[0] = layers[0].to(dev)
    dtype = model_dtype if not args.use_fp32 else torch.float32
    traincast = torch.cuda.amp.autocast if not args.use_fp32 else nullcontext

    # step 2: init dataset
    fp_train_inps = []
    
    # step 3: catch the input of thefirst layer 
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            fp_train_inps.append(inp)
            layer_kwargs_list.append(kwargs)
            raise ValueError
    
    # step 3.1: catch the input of training set
    layers[0] = Catcher(layers[0])
    inputs, vision_mask, caption_mask = process_input(prompt_inputs, prompt_kwargs)

    model.to_cuda()
    iters = args.train_size//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            try:
                model(
                    inputs_embeds=inputs["inputs_embeds"][i].unsqueeze(0), 
                    attention_mask=inputs["attention_mask"][i].unsqueeze(0),
                    labels=inputs["labels"][i].unsqueeze(0),
                    use_cache=inputs["use_cache"],
                )
            except ValueError: # work with early exit
                pass

    model.to_cpu()            
    layers[0] = layers[0].module
    layer_kwargs["use_cache"] = False
    
    # step 4: move embedding layer and first layer to cpu
    layers[0] = layers[0].cpu()
    move_embed(model.model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    # step 5: copy fp input as the quant input, they are same at the first layer
    quant_train_inps = copy.deepcopy(fp_train_inps)
    
    
    # step 6: start training    
    loss_func = get_recon_loss(args.loss_type) 
    # for block_index in range(1):
    for block_index in range(len(layers)):
        logger.info(f"=== Start quantize blocks {block_index}===")
        qlayer = layers[block_index].to(dev)
        
        qlayer.to(dev)
        # obtain output of full-precision model
        if args.epochs > 0:
            set_quant_state(qlayer,weight_quant=False,act_quant=False)
            update_dataset(qlayer,fp_train_inps,dev,layer_kwargs_list)
            
            
        # activate quantization
        if args.quant_type == 'weight_only':
            set_quant_state(qlayer,weight_quant=True,act_quant=False)
        else:
            set_quant_state(qlayer,weight_quant=True,act_quant=True)  
        
        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # fp32 is also required for AMP training
            # create optimizer
            assert args.clip_lr > 0 or args.weight_lr > 0
            set_clip_parameters(qlayer,args.clip_lr > 0)
            set_weight_lora_parameters(qlayer,args.weight_lr > 0)
            param = []
            if args.clip_lr > 0:
                param.append({"params":clip_parameters(qlayer),"lr":args.clip_lr})
            if args.weight_lr > 0:
                param.append({"params":weight_lora_parameters(qlayer),"lr":args.weight_lr})
                
                        
            optimizer = torch.optim.AdamW(param, weight_decay=args.wd)
            loss_scaler = NativeScalerWithGradNormCount()
            trainable_number = trainable_parameters_num(qlayer, args)
            logger.info(f"trainable parameter number: {trainable_number/1e6}M")
            # import pdb;pdb.set_trace()
            
            if not args.progressive_training:
                total_training_iteration = args.epochs * args.train_size / args.batch_size
                lr_schedule = CustomLRSchedule(args, total_training_iteration)
                for epoch in range(args.epochs):
                    # training
                    start_time = time.time()
                    train_loss, gradient_norm = train_one_epoch(qlayer, layer_kwargs_list,
                        loss_scaler, loss_func, lr_schedule, optimizer, dev, traincast, quant_train_inps, fp_train_inps, block_index, args)
                    
                    logger.info(f"blocks {block_index} epoch {epoch} train_loss:{train_loss} norm:{gradient_norm:.8f} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2:.2f}MB time {time.time()-start_time:.1f} ")
            optimizer.zero_grad()
            del optimizer

        
        if args.epochs>0:
            # update inputs of quantization model
            update_dataset(qlayer,quant_train_inps,dev,layer_kwargs_list, is_token_fusion = True, quant_type = args.quant_type)# , is_token_fusion = True, quant_type = args.quant_type
        if args.reduce_memory:
            qlayer.to(model_dtype)
            # qlayer.half()   
        
        # move to cpu
        layers[block_index] = qlayer.to("cpu")

    torch.cuda.empty_cache()
    gc.collect()                    
    # model.config.use_cache = use_cache
    return model.model

