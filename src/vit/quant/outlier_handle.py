import numpy as np
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import random
from functools import reduce
from quant.quant_modules import QuantLinear,QuantMatMul, QuantConv2d, QuantScalableLinear
from quant.quant_model import set_quant_state, model_bit_refactor, set_quant_state_layers, layers_bit_refactor
from quant.quant_vision_transformer_block import QViTBlock
from quant.quantizer import lp_loss
from quant.set_quant_params import set_weight_quantize_params, set_act_quantize_params
from utils.data_utils import save_inp_oup_data
import warnings
 
warnings.filterwarnings("ignore")
MB = 1024.0 * 1024.0

def outlier_handler_block(model, block, mixed_inputs = None, args = None):
    
    # set model in lowest bit-width
    model_bit_refactor(model=model,bit= args.wbits)

    set_weight_quantize_params(block)
    set_act_quantize_params(block, mixed_inputs, True, True)
    
    if 'swin' in args.model: # for swin 
        module_dict = {}
        for name, module in block.named_modules():
            module_dict[name] = module
            idx = name.rfind('.')
            if idx == -1:
                idx = 0
            father_name = name[:idx]
            if father_name in module_dict:
                father_module = module_dict[father_name]
            else:
                raise RuntimeError(f"father module {father_name} not found")
           
            if 'norm1' in name or 'norm2' in name or 'norm' in name:
                if 'norm1' in name:
                    next_module = father_module.attn.qkv
                elif 'norm2' in name:
                    next_module = father_module.mlp.fc1
                else:
                    next_module = father_module.reduction
                
                act_delta = next_module.act_quantizer.delta_list[0].reshape(-1)
                act_zero_point = next_module.act_quantizer.zero_point_list[0].reshape(-1)
                act_min = -act_zero_point * act_delta
                
                target_delta = torch.mean(act_delta)
                target_zero_point = torch.mean(act_zero_point)
                target_min = -target_zero_point * target_delta

                r = act_delta / target_delta
                b = act_min / r - target_min

                module.weight.data = module.weight.data / r
                module.bias.data = module.bias.data / r - b

                next_module.weight.data = next_module.weight.data * r
                if next_module.bias is not None:
                    next_module.bias.data = next_module.bias.data + torch.mm(next_module.weight.data, b.reshape(-1,1)).reshape(-1)
                else:
                    next_module.bias = nn.Parameter(torch.Tensor(next_module.weight.shape[0]))
                    next_module.bias.data = torch.mm(next_module.weight.data, b.reshape(-1,1)).reshape(-1)

                next_module.act_quantizer.channel_wise = False

    else: # for vit-based
        for name, module in block.named_modules():
            if 'norm1' in name or 'norm2' in name or 'norm' in name:
                if 'norm1' in name:
                    next_module = block.attn.qkv
                elif 'norm2' in name:
                    next_module = block.mlp.fc1
                else:
                    next_module = block.reduction

                act_delta = next_module.act_quantizer.delta_list[0].reshape(-1)
                act_zero_point = next_module.act_quantizer.zero_point_list[0].reshape(-1)
                act_min = -act_zero_point * act_delta
                
                target_delta = torch.mean(act_delta)
                target_zero_point = torch.mean(act_zero_point)
                target_min = -target_zero_point * target_delta

                r = act_delta / target_delta
                b = act_min / r - target_min

                module.weight.data = module.weight.data / r
                module.bias.data = module.bias.data / r - b

                next_module.weight.data = next_module.weight.data * r
                if next_module.bias is not None:
                    next_module.bias.data = next_module.bias.data + torch.mm(next_module.weight.data, b.reshape(-1,1)).reshape(-1)
                else:
                    next_module.bias = nn.Parameter(torch.Tensor(next_module.weight.shape[0]))
                    next_module.bias.data = torch.mm(next_module.weight.data, b.reshape(-1,1)).reshape(-1)

                next_module.act_quantizer.channel_wise = False
 
    set_weight_quantize_params(block)
    set_act_quantize_params(block, mixed_inputs, True, True)
    
def block_handle(model, block, cali_data,
                        batch_size: int = 32, 
                        keep_gpu: bool = True, 
                        input_prob: float = 0.5,
                        args = None):
   
    # block.set_quant_state
    for m in block.modules():
        if isinstance(m, (QuantLinear, QuantMatMul, QuantScalableLinear)):
            m.set_quant_state(True, True)
            
    '''get input and set scale'''
    quant_inps, fp_inp, fp_out = mfm(model, block, cali_data, batch_size, keep_gpu, input_prob, args = args)
    outlier_handler_block(model, block, mixed_inputs=(0.5 * quant_inps[0][:32] + 0.5 * quant_inps[1][:32]), args=args)
    
    print(f"handle outlier done")
    
def mfm(model, layer, input_data, bs, keep_gpu, input_prob, args):
        cached_inps_quant_list1 = []
        cached_inps_quant_list2 = []
        cached_outs_fp = None
        cached_inps_fp = None

        groups1 = list()
        # groups1.append(random.choice(args.group_H))
        # groups1.append(random.choice(args.group_M))
        groups1.append(random.choice(args.group_L))

        for i in groups1:
            model_bit_refactor(model=model, bit = i)
            
            cached_inps_quant, cached_outs_fp =save_inp_oup_data(model, layer, input_data, bs, 
                                                                 input_prob=input_prob, keep_gpu=keep_gpu)
            
            cached_inps_quant_list1.append(cached_inps_quant[0])
            cached_inps_fp = cached_inps_quant[1]
        return (cached_inps_quant[0], cached_inps_quant[0]), cached_inps_fp, cached_outs_fp

def layer_handle(model, block, cali_data,
                        batch_size: int = 32, 
                        input_prob: float = 0.5, 
                        keep_gpu: bool = True, 
                        args = None):

    # block.set_quant_state
    block.set_quant_state(True, True)
    device = 'cuda'
    quant_inps, fp_inp, fp_out = mfm(model, block, cali_data, batch_size, keep_gpu, input_prob, args = args)
    if block.is_last_layer:
        set_weight_quantize_params(block)
        set_act_quantize_params(block, (0.5*quant_inps[0][:32].to(device)+
                                        0.5* quant_inps[1][:32].to(device)),
                                    True, True)
        return