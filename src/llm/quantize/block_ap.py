import torch
import random
import torch.nn as nn
import quantize.int_linear_fake as int_linear_fake
import quantize.int_linear_real as int_linear_real
from quantize.recon_loss import get_recon_loss
from quantize.int_linear_fake import QuantLinear

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
import os

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
    
def huber_loss(input, target, delta = 1.0):
    diff = torch.abs(input - target)
    diff_flatten = diff.flatten()
    k= diff_flatten.numel()// 10
    values, indices = torch.topk(diff_flatten, k)
    delta = values[-1]
    cond = diff< delta
    loss = torch.where(cond,(input-target).abs().pow(2),(input-target).abs().pow(1))
    return loss.sum(1).mean()


@torch.no_grad()
def update_dataset(layer, dataset, dev, attention_mask, position_ids, args, is_token_fusion = False, 
                   quant_type = 'weight_only',topk_token = 0.5, bit_allc=None):
    with torch.no_grad(): 
        with torch.cuda.amp.autocast():
            if args.selective_merge:
                for index, inps in enumerate(dataset):
                    inps = inps.to(dev)
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
                                _.append(layer(inps[i:i+1], attention_mask=attention_mask,position_ids=position_ids)[0])
                        else:
                            bit_groups = [[4],[5,6,],[7,8]] # ,[5,6,],[7,8]
                            for group_index, bit_choices in zip(range(len(bit_groups)), bit_groups):
                                b = random.choice(bit_choices)
                                model_bit_refactor(layer, b)
                                _.append(layer(inps[i:i+1], attention_mask=attention_mask,position_ids=position_ids)[0])
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
                    dataset.update_data(index,new_data)

            elif args.uniform_fusion:
                for index, inps in enumerate(dataset):
                    inps = inps.to(dev)
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
                                _.append(layer(inps[i:i+1], attention_mask=attention_mask,position_ids=position_ids)[0])
                        else:
                            bit_groups = [[4],[5,6,],[7,8]] # ,[5,6,],[7,8]
                            for group_index, bit_choices in zip(range(len(bit_groups)), bit_groups):
                                b = random.choice(bit_choices)
                                model_bit_refactor(layer, b)
                                _.append(layer(inps[i:i+1], attention_mask=attention_mask,position_ids=position_ids)[0])
                        if quant_type == 'weight_only':
                            mfm_lambda = 0.25
                            if torch.isnan((_[0] + _[1] + _[2] + _[3]) * mfm_lambda).any():
                                import pdb 
                                pdb.set_trace() 
                            batch_feature.append((_[0] + _[1] + _[2] + _[3]) * mfm_lambda)
                        else:
                            mfm_lambda = [0.4,0.3,0.3]
                            batch_feature.append(_[0] * mfm_lambda[0] + _[1] * mfm_lambda[1]+ _[2] * mfm_lambda[2])
                    new_data = torch.cat([x for x in batch_feature]).to('cpu')
                    dataset.update_data(index,new_data)

            else:
                for index, inps in enumerate(dataset):
                    inps = inps.to(dev)
                    if len(inps.shape)==2:
                        inps = inps.unsqueeze(0)
                    new_data = layer(inps, attention_mask=attention_mask,position_ids=position_ids)[0].to('cpu')
                    dataset.update_data(index,new_data)
    torch.cuda.empty_cache()
    train_utils.cleanup_memory()
            
def train_one_epoch(qlayer, attention_mask, position_ids,
                      loss_scaler, loss_func, lr_schedule, optimizer, dev, traincast,
                      quant_inps, fp_inps_with_fp, epoch, block_index, args):
    loss_list = []
    norm_list = []
    for index in range(len(quant_inps)): # batch-wise
        with traincast():
            input = quant_inps[index].to(dev)
            label = fp_inps_with_fp[index].to(dev)
            loss = None
            cached_out = None
            if args.quant_type == 'weight_only': # for weight only use 2-8, 4 groups
                bit_groups = [[7,8],[5,6],[3,4],[2]] # ,[3,4,],[5,6,],[7,8]
                for group_index, bit_choices in zip(range(len(bit_groups)), bit_groups):
                    bit = random.choice(bit_choices)
                    model_bit_refactor(qlayer, bit)
                    quant_out = qlayer(input, attention_mask=attention_mask,position_ids=position_ids)[0]
                    if group_index == 0:# high group
                        if args.mae_loss:
                            loss = lp_loss(quant_out, label, p=1)
                        elif args.huber_loss:
                            loss = huber_loss(quant_out, label)
                        else:
                            loss = lp_loss(quant_out, label)
                    else:
                        if args.mae_loss:
                            mae_loss = lp_loss(quant_out, label, p=1)
                            loss += mae_loss
                        elif args.huber_loss:
                            h_loss = huber_loss(quant_out, label)
                            loss += h_loss
                        else:
                            mse_loss = lp_loss(quant_out, label)
                            loss += mse_loss

                    if not math.isfinite(loss.item()):
                        import pdb;pdb.set_trace()
            elif args.quant_type == 'weight_act':       
                bit_groups = [[7,8],[5,6,],[4]] #
                for group_index, bit_choices in zip(range(len(bit_groups)), bit_groups):
                    bit = random.choice(bit_choices)
                    model_bit_refactor(qlayer, bit)
                    quant_out = qlayer(input, attention_mask=attention_mask,position_ids=position_ids)[0]
                    if group_index == 0:# high group
                        if args.mae_loss:
                            loss = lp_loss(quant_out, label, p=1)
                        elif args.huber_loss:
                            loss = huber_loss(quant_out, label)
                        else:
                            loss = lp_loss(quant_out, label)
                    else:
                        if args.mae_loss:
                            mae_loss = lp_loss(quant_out, label, p=1)
                            loss += mae_loss
                        elif args.huber_loss:
                            h_loss = huber_loss(quant_out, label)
                            loss += h_loss
                        else:
                            mse_loss = lp_loss(quant_out, label)
                            loss += mse_loss
                    if not math.isfinite(loss.item()):
                        import pdb;pdb.set_trace()      

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
def eval_one_epoch(qlayer, attention_mask, position_ids,
                      loss_func, dev, traincast,
                      quant_inps, fp_inps_with_fp, args):
    loss_list_low = []
    loss_list_high = []
    for index in range(len(quant_inps)):
        with traincast():
            input = quant_inps[index].to(dev)
            label = fp_inps_with_fp[index].to(dev)
            if args.quant_type == 'weight_only': # for weight only use 2-8, 4 groups
                bit_groups = [[2],[7,8]]
                for group_index, bit_choices in zip(range(len(bit_groups)), bit_groups):
                    bit = random.choice(bit_choices)
                    model_bit_refactor(qlayer, bit)
                    quant_out = qlayer(input, attention_mask=attention_mask,position_ids=position_ids)[0]
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
                    quant_out = qlayer(input, attention_mask=attention_mask,position_ids=position_ids)[0]
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
            if len(optimizer.param_groups) == 1:
                optimizer.param_groups[0]['lr'] = self.weight_scheduler.get_lr()[0]
            else:
                optimizer.param_groups[self.weight_index]['lr'] = self.weight_scheduler.get_lr()[0]

class GDLOSSLRSchedule(object):
    def __init__(self, args, total_iter) -> None:
        if args.weight_lr > 0:
            empty_optimizer = torch.optim.AdamW([torch.tensor(0)], lr=args.weight_lr)
            self.weight_scheduler = CosineAnnealingLR(empty_optimizer, T_max=total_iter, eta_min=args.weight_lr/args.min_lr_factor)  
        else:
            self.weight_scheduler = None
    def step(self, optimizer):
        if self.weight_scheduler is not None:
            self.weight_scheduler.step()
            optimizer.param_groups[0]['lr'] = self.weight_scheduler.get_lr()[0]        
             
def block_ap(
    model,
    args,
    trainloader,
    valloader,
    logger=None,
):
    logger.info("Starting ...")
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model_dtype = next(model.parameters()).dtype
    # step 1: move embedding layer and first layer to target device, only suppress llama models now
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = model_dtype if not args.use_fp32 else torch.float32
    traincast = torch.cuda.amp.autocast if not args.use_fp32 else nullcontext

    # step 2: init dataset
    fp_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                model.config.hidden_size, args.batch_size, dtype, cache_path=args.cache_dir)
    
    # step 3: catch the input of thefirst layer 
    class Catcher(nn.Module):
        def __init__(self, module, dataset):
            super().__init__()
            self.module = module
            self.dataset = dataset
            self.index = 0
            self.attention_mask = None
            self.position_ids = None

        def forward(self, inp, **kwargs):
            self.dataset.update_data(self.index, inp.squeeze(0).to('cpu'))
            self.index += 1
            if self.attention_mask is None:
                self.attention_mask = kwargs["attention_mask"]
            if self.position_ids is None:
                self.position_ids = kwargs["position_ids"]
            raise ValueError
    
    # step 3.1: catch the input of training set
    layers[0] = Catcher(layers[0],fp_train_inps)
    iters = len(trainloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([trainloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    position_ids = layers[0].position_ids
    attention_mask = layers[0].attention_mask
    attention_mask = attention_mask.to(dtype) if attention_mask is not None else None
    layers[0] = layers[0].module
    
    # step 4: move embedding layer and first layer to cpu
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()
    
    # step 5: copy fp input as the quant input, they are same at the first layer
    quant_train_inps = copy_block_dataset(fp_train_inps)
    
    # step 6: start training    
    loss_func = get_recon_loss(args.loss_type) 
    for block_index in range(len(layers)):
        logger.info(f"=== Start quantize blocks {block_index}===")
        qlayer = layers[block_index].to(dev)     
        
        qlayer.to(dev)
        # obtain output of full-precision model
        if args.epochs > 0:
            set_quant_state(qlayer,weight_quant=False,act_quant=False)
            update_dataset(qlayer,fp_train_inps,dev,attention_mask,position_ids, args)
            
            
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
            lora_param = []
            if args.clip_lr > 0:
                param.append({"params":clip_parameters(qlayer),"lr":args.clip_lr})
            if args.weight_lr > 0:
                param.append({"params":weight_lora_parameters(qlayer),"lr":args.weight_lr})
                        
            optimizer = torch.optim.AdamW(param, weight_decay=args.wd)
            total_training_iteration = args.epochs * args.train_size / args.batch_size             
            loss_scaler = NativeScalerWithGradNormCount()
            trainable_number = trainable_parameters_num(qlayer, args)
            logger.info(f"trainable parameter number: {trainable_number/1e6}M")
            
            lr_schedule = CustomLRSchedule(args, total_training_iteration)
            for epoch in range(args.epochs):
                # training
                start_time = time.time()
                train_loss, gradient_norm = train_one_epoch(
                    qlayer, attention_mask, position_ids,
                    loss_scaler, loss_func, lr_schedule, 
                    optimizer, dev, traincast, quant_train_inps, fp_train_inps, epoch, block_index, args)
                
                logger.info(f"blocks {block_index} epoch {epoch} train_loss:{train_loss} norm:{gradient_norm:.8f} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2:.2f}MB time {time.time()-start_time:.1f} ")
                mid_time = time.time()
            optimizer.zero_grad()
            del optimizer
    
        if args.epochs>0:
            # update inputs of quantization model
            update_dataset(qlayer,quant_train_inps,dev,attention_mask,position_ids,args,
                           is_token_fusion = args.token_fusion, quant_type = args.quant_type
                          )
        if args.reduce_memory:
            qlayer.to(model_dtype)
            # qlayer.half()   
        
        # move to cpu
        layers[block_index] = qlayer.to("cpu")

    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

