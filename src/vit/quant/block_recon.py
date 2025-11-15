import warnings
import torch
import gc
import random
import torch.nn as nn
import torch.nn.functional as F
from quant.quant_modules import QuantLinear,QuantMatMul, QuantScalableLinear
from quant.quant_model import set_quant_state_stage, stage_bit_refactor
from quant.quantizer import lp_loss
from utils.data_utils import save_inp_oup_data

warnings.filterwarnings("ignore")
MB = 1024.0 * 1024.0

def cal_entropy(attn):
    return -1 * torch.sum((attn * torch.log(attn)), dim=-1).mean()


def learnable_parameters_swin(stage, args = None):
     
    lora_para, scale_para, clip_para, alphas_para = [], [], [], []
    for name, module in stage.named_modules():
        '''weight'''
        if isinstance(module, QuantScalableLinear):
            lora_para += [module.Bu]
            lora_para += [module.Bd]
            
            if module.weight_quantizer.lwc:
                clip_para += [module.weight_quantizer.upbound_factor]
                clip_para += [module.weight_quantizer.lowbound_factor]
        
            
        '''activation'''
        if isinstance(module, (QuantLinear, QuantScalableLinear)):
            module.act_quantizer.delta_list = torch.nn.Parameter(module.act_quantizer.delta_list)
            scale_para += [module.act_quantizer.delta_list]

        if isinstance(module,QuantMatMul):
            if not module.disable_quantizer_a:
                module.quantizer_A.delta_list = torch.nn.Parameter(module.quantizer_A.delta_list)
                scale_para += [module.quantizer_A.delta_list]
            
            module.quantizer_B.delta_list = torch.nn.Parameter(module.quantizer_B.delta_list)
            scale_para += [module.quantizer_B.delta_list]

    return lora_para, scale_para, clip_para, alphas_para

def learnable_parameters(sublayer, args = None):
     
    lora_para, scale_para, clip_para, alphas_para = [], [], [], []
    '''weight'''
    
    for name, module in sublayer.named_modules():   
        if isinstance(module, QuantScalableLinear):
            lora_para += [module.Bu]
            lora_para += [module.Bd]
            
            if module.weight_quantizer.lwc:
                clip_para += [module.weight_quantizer.upbound_factor]
                clip_para += [module.weight_quantizer.lowbound_factor]
        
        '''activation'''
        if isinstance(module, (QuantLinear, QuantScalableLinear)):
            module.act_quantizer.delta_list = torch.nn.Parameter(module.act_quantizer.delta_list)
            scale_para += [module.act_quantizer.delta_list]

        if isinstance(module,QuantMatMul):
            if not module.disable_quantizer_a:
                module.quantizer_A.delta_list = torch.nn.Parameter(module.quantizer_A.delta_list)
                scale_para += [module.quantizer_A.delta_list]
            
            module.quantizer_B.delta_list = torch.nn.Parameter(module.quantizer_B.delta_list)
            scale_para += [module.quantizer_B.delta_list]

        # if isinstance(module, QuantScalableLinear) and module.quant_mode == 'lora_adaround':
        #     alphas_para += [module.alpha]

    return lora_para, scale_para, clip_para, alphas_para

def rec_model(q_model, cali_data, batch_size: int = 16, iters: int = 500, 
                lr_weight:  float = 4e-5, lr_scale: float = 4e-5, input_prob: float = 0.5, 
                keep_gpu: bool = True, logger = None, args = None):

    total_batch = int(cali_data.size(0) / batch_size)
    layers = q_model.layers if 'swin' in args.model else q_model.blocks
    swin_flag = True if 'swin' in args.model else False
    # swin layers as stage
    if swin_flag:
        cached_inps_quant, cached_outs_fp =save_inp_oup_data(q_model, layers[0].blocks[0], cali_data, batch_size, input_prob=input_prob, keep_gpu=keep_gpu)
    else:
        cached_inps_quant, cached_outs_fp =save_inp_oup_data(q_model, layers[0], cali_data, batch_size, input_prob=input_prob, keep_gpu=keep_gpu)
    
    fp_inps = cached_inps_quant[1]
    quant_inps = cached_inps_quant[0]
    if swin_flag:
        for cur_stage, stage in zip(range(len(layers)), layers): # optimize swin stage by stage
            ## learnable parameters
            lora_weight_para, scale_para, clip_para, alpha_para = learnable_parameters_swin(stage)
            ## opt setting
            optimizer = torch.optim.Adam([{"params":lora_weight_para, "lr":lr_weight},
                                          {"params":scale_para, "lr":lr_scale},
                                          {"params":clip_para, "lr":args.clip_lr},
                                          {"params":alpha_para, "lr":args.loraa_lr}])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * total_batch, eta_min=0.)
            for e in range(args.epochs):
                h_loss_records = []
                m_loss_records = []
                l_loss_records = []    
                for id_batch in range(total_batch):
                    cur_inp = quant_inps[id_batch * batch_size:(id_batch+1)*batch_size]
                    cur_fp_inps = fp_inps[id_batch * batch_size:(id_batch+1)*batch_size]
                    
                    set_quant_state_stage(stage, False, False)
                    fp_out = stage(cur_fp_inps)
                    set_quant_state_stage(stage, True, True)
                    optimizer.zero_grad()
                    act_cached = None
                    for bit_g in range(3):
                        if bit_g == 0: # highest group
                            temp = random.choice(args.group_H)
                            stage_bit_refactor(stage, temp) 
                            output = stage(cur_inp)
                            if args.mae:
                                rec_loss = lp_loss(output, fp_out, p=1)
                            else:
                                rec_loss = lp_loss(output, fp_out)
                            h_loss_records.append(rec_loss.item())
                        elif bit_g == 2: # lowest group
                            temp = random.choice(args.group_L)
                            stage_bit_refactor(stage, temp)
                            output = stage(cur_inp)
                            if args.mae:
                                rec_loss = lp_loss(output, fp_out, p=1)
                            else:
                                rec_loss = lp_loss(output, fp_out)
                            l_loss_records.append(rec_loss.item())
                        else:
                            temp = random.choice(args.group_M)
                            stage_bit_refactor(stage, temp)
                            output = stage(cur_inp)
                            if args.mae:
                                rec_loss = lp_loss(output, fp_out, p=1)
                            else:
                                rec_loss = lp_loss(output, fp_out)
                            m_loss_records.append(rec_loss.item())
                        
                        loss = rec_loss
                        loss.backward(retain_graph=True)        
                    
                    optimizer.step()
                    scheduler.step()
                # loss record
                h_loss = sum(h_loss_records) / len(h_loss_records)
                m_loss = sum(m_loss_records) / len(m_loss_records)
                l_loss = sum(l_loss_records) / len(l_loss_records)
                # logging
                # if (i + 1) % (iters // 4) == 0:
                logger.info("[Stage-{}] [Epochs: {}/{}] [high bit loss:{:.4f}] | [middle bit loss:{:.4f}] | [low bit loss:{:.4f}] ".format(cur_stage+1,e+1,args.epochs, h_loss, m_loss, l_loss)) 

            torch.cuda.empty_cache()

            set_quant_state_stage(stage, False, False)
            fp_cached_batches = []
            with torch.no_grad():
                for i in range(total_batch):
                    fp_cached_batches.append(stage(fp_inps[i * batch_size:(i+1)*batch_size]))
            
            fp_inps = torch.cat([x for x in fp_cached_batches])
            del fp_cached_batches 
            gc.collect()
            torch.cuda.empty_cache()
            
            mixer_level = args.mixer_level
            quant_inps = feature_mixer(stage, quant_inps, batch_size, level=mixer_level, args=args)             
    else:
        for cur_index, block in zip(range(len(layers)), layers): # optimize vit/deit block by block
            ## learnable parameters 
            lora_weight_para, scale_para, clip_para, alpha_para = learnable_parameters(sublayer=block)
            ## opt setting
            optimizer = torch.optim.Adam([{"params":lora_weight_para, "lr":lr_weight},
                                          {"params":scale_para, "lr":lr_scale},
                                          {"params":clip_para, "lr":args.clip_lr},
                                          {"params":alpha_para, "lr":args.loraa_lr}])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * total_batch, eta_min=0.)
            loss_criterion = nn.MSELoss().cuda()

            for e in range(args.epochs):
                h_loss_records = []
                m_loss_records = []
                l_loss_records = []
                for id_batch in range(total_batch):
                    cur_inp = quant_inps[id_batch * batch_size:(id_batch+1)*batch_size]
                    cur_fp_inps = fp_inps[id_batch * batch_size:(id_batch+1)*batch_size]
                    
                    block.set_quant_state(False, False)

                    fp_out = block(cur_fp_inps)
                
                    block.set_quant_state(True, True)
        
                    optimizer.zero_grad()
                    act_cached = None   
                    for bit_g in range(3):
                        if bit_g == 0: # highest group
                            temp = random.choice(args.group_H) 
                            block.bit_refactor(temp)
                            output = block(cur_inp)
                            if args.mae:
                                rec_loss = lp_loss(output, fp_out, p=1)
                            else:
                                rec_loss = lp_loss(output, fp_out)
                            h_loss_records.append(rec_loss.item())
                    
                        elif bit_g == 2: # lowest group
                            temp = random.choice(args.group_L)
                            block.bit_refactor(temp)

                            output = block(cur_inp)
                            rec_loss = loss_criterion(output, fp_out)
                            if args.mae:
                                rec_loss = lp_loss(output, fp_out, p=1)
                            else:
                                rec_loss = lp_loss(output, fp_out)
                            l_loss_records.append(rec_loss.item())
                        elif bit_g == 1:
                            temp = random.choice(args.group_M)
                            block.bit_refactor(temp)

                            output = block(cur_inp)
                            rec_loss = loss_criterion(output, fp_out)
                            if args.mae:
                                rec_loss = lp_loss(output, fp_out, p=1)                                     
                            else:
                                rec_loss = lp_loss(output, fp_out)
                            m_loss_records.append(rec_loss.item())

                        loss = rec_loss
                        loss.backward(retain_graph=True)
                    
                    optimizer.step()
                    scheduler.step()
                # loss record
                h_loss = sum(h_loss_records) / len(h_loss_records)
                m_loss = sum(m_loss_records) / len(m_loss_records)
                l_loss = sum(l_loss_records) / len(l_loss_records)

                logger.info("[Block-{}] [Epochs: {}/{}] [high bit loss:{:.4f}] | [middle bit loss:{:.4f}] | [low bit loss:{:.4f}] ".format(cur_index,e+1,args.epochs, h_loss, m_loss, l_loss)) 
   
            torch.cuda.empty_cache()
            
            block.set_quant_state(False, False)
            fp_cached_batches = []
            with torch.no_grad():
                for i in range(total_batch):
                    fp_cached_batches.append(block(fp_inps[i * batch_size:(i+1)*batch_size]))
            fp_inps = torch.cat([x for x in fp_cached_batches])
            del fp_cached_batches 
            gc.collect()
            torch.cuda.empty_cache()
            
            # bit_allc = []
            mixer_level = args.mixer_level
            
            quant_inps = feature_mixer(block, quant_inps, batch_size, level=mixer_level, args=args)
            cur_index += 1
   
def feature_mixer(layer, inps, batch_size, level, args, bit_allc = None):
    
    if level == "Random-Selection":
        quant_cached_batches = []
        if 'swin' in args.model:
            set_quant_state_stage(layer, True, True)
        else:
            layer.set_quant_state(True, True)
        with torch.no_grad():
            for i in range(inps.size(0)):
                _ = []
                for bit in [4,5,6,7,8]:
                    if 'swin' in args.model:
                        stage_bit_refactor(layer, bit)
                    else:
                        layer.bit_refactor(bit)
                    _.append(layer(inps[i:i+1]))
                try:
                    B, N, C = _[0].shape
                except ValueError:
                    print(_[0].shape)
                
                for index in range(len(_)):
                    _[index] = _[index].reshape(-1, _[index].size(-1))
                
                f = []
                if bit_allc is None:
                    for t in range(N):
                        b = random.choice([4,5,6,7,8])
                        if b == 4:
                            f.append(_[0][t])
                        elif b == 5:
                            f.append(_[1][t])
                        elif b == 6:
                            f.append(_[2][t])
                        elif b == 7:
                            f.append(_[3][t])
                        elif b == 8:
                            f.append(_[4][t]) 
                else:
                    for t, b in zip(range(N), bit_allc):
                        if b == 4:
                            f.append(_[0][t])
                        elif b == 5:
                            f.append(_[1][t])
                        elif b == 6:
                            f.append(_[2][t])
                        elif b == 7:
                            f.append(_[3][t])
                        elif b == 8:
                            f.append(_[4][t]) 
                f = torch.cat([x for x in f])
                f = f.reshape(B, N, C)
                quant_cached_batches.append(f)
        
        quant_inps = torch.cat([x for x in quant_cached_batches])
        del quant_cached_batches 
        gc.collect()
        torch.cuda.empty_cache()
        return quant_inps  
      
    elif level == "Uniform-Fusion":
        quant_cached_batches = []
        if 'swin' in args.model:
            set_quant_state_stage(layer, True, True)
        else:
            layer.set_quant_state(True, True)
        total_batch = int(inps.size(0) / batch_size)
        with torch.no_grad(): 
            for t in range(total_batch):
                    bits = list()
                    cached_inps_quant = list()
                    bits.append(random.choice(args.group_H))
                    bits.append(random.choice(args.group_M))
                    bits.append(random.choice(args.group_L))
                    for b in bits:
                        if 'swin' in args.model:
                            stage_bit_refactor(layer, b)
                        else:
                            layer.bit_refactor(b)
                        cached_inps_quant.append(layer(inps[t * batch_size:(t+1)*batch_size]))
                    mfm_lambda = args.MFM_PARAM
                    quant_cached_batches.append(
                            cached_inps_quant[0] * mfm_lambda[0]
                            + cached_inps_quant[1] * mfm_lambda[1]
                            + cached_inps_quant[2] * mfm_lambda[2]
                    )
            quant_inps = torch.cat([x for x in quant_cached_batches])
            del quant_cached_batches 
            gc.collect()
            torch.cuda.empty_cache()
            return quant_inps


    elif level == 'Selective-Merge':
        quant_cached_batches = []
        if 'swin' in args.model:
            set_quant_state_stage(layer, True, True)
        else:
            layer.set_quant_state(True, True)
        with torch.no_grad():
            for i in range(inps.size(0)):
                _ = []
                bits = list()
                bits.append(random.choice(args.group_H))
                bits.append(random.choice(args.group_M))
                bits.append(random.choice(args.group_L))
                for b in bits:
                    if 'swin' in args.model:
                        stage_bit_refactor(layer, b)
                    else:
                        layer.bit_refactor(b)
                    _.append(layer(inps[i:i+1]))

                B, N, C = _[0].shape
                
                for index in range(len(_)):
                    _[index] = _[index].reshape(-1, _[index].size(-1))
                
                f = []
                k = int((args.topk_token) * N)
                similarity, top_k_token = cosine_similarity(_[0], _[-1], top_k=k)
                
                for t in range(N):
                    if t in top_k_token.tolist():
                        f.append(_[0][t])
                    else:
                        mfm_lambda = args.MFM_PARAM
                        f.append(_[0][t] * mfm_lambda[0] + _[1][t] * mfm_lambda[1]+ _[2][t] * mfm_lambda[2])
                    
                f = torch.cat([x for x in f])
                f = f.reshape(B, N, C)
                quant_cached_batches.append(f)
        
        quant_inps = torch.cat([x for x in quant_cached_batches])
        del quant_cached_batches 
        gc.collect()
        torch.cuda.empty_cache()
        return quant_inps
    
def cosine_similarity(input1:torch.Tensor, input2:torch.Tensor, top_k = 0):
    normalized_input1 = input1 / input1.norm(dim=-1, keepdim=True)
    normalized_input2 = input2 / input2.norm(dim=-1, keepdim=True)
    similarity = (normalized_input1 * normalized_input2).sum(dim=-1)
    if top_k >0 :
        _, idx = torch.topk(similarity, k=top_k)
        return similarity, idx
    else:
        return similarity


    
                
                
      
                
            
        





