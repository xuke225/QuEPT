import random
import torch
import torch.nn as nn
from copy import deepcopy
from .quant_vision_transformer_block import QViTBlock, QSwinBlock
from .quant_modules import QuantConv2d, QuantLinear, QuantMatMul, QuantScalableLinear
from .quantizer import UniformQuantizer_Act, UniformQuantizer_Weight
from timm.models.vision_transformer import Block
from timm.models.swin_transformer import SwinTransformerBlock

def QViT(model: nn.Module,args):
    for name, child_module in model.named_children():
        if isinstance(child_module,Block):
            setattr(
                model,
                name,
                QViTBlock(
                    child_module, args
                ),
            )
        elif isinstance(child_module, nn.Conv2d):
            setattr(
                model,
                name,
                QuantConv2d(
                    child_module, args.weight_quant_params, args.act_quant_params
                ),
            )
        elif isinstance(child_module, nn.Linear):
            setattr(
                model,
                name,
                QuantLinear(
                    child_module, args.weight_quant_params, args.act_quant_params
                ),
            )
        else:
            QViT(
                child_module, args
            )
    set_quant_state(model, False, False)
    return model

def QSwinViT(model: nn.Module,args):

    for name, child_module in model.named_children():
        if isinstance(child_module, SwinTransformerBlock):
            setattr(
                model,
                name,
                QSwinBlock(
                    child_module, args
                ),
            )
            # pass
        elif isinstance(child_module, nn.Conv2d):
            setattr(
                model,
                name,
                QuantConv2d(
                    child_module, args.weight_quant_params, args.act_quant_params
                ),
            )
        elif isinstance(child_module, nn.Linear):
            if 'reduction' in name:
                reduction_quant_params = deepcopy(args.act_quant_params)
                reduction_quant_params['channel_wise'] = True
                setattr(
                    model,
                    name,
                    QuantScalableLinear(
                        child_module, args.weight_quant_params, reduction_quant_params
                    ),
                )        
            else:
                setattr(
                    model,
                    name,
                    QuantLinear(
                        child_module, args.weight_quant_params, args.act_quant_params
                    ),
                )
        else:
            QSwinViT(
                child_module, args
            )

    return model

def set_quant_state(model, weight_quant=False, act_quant=False):
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantMatMul, QuantScalableLinear)):
            m.set_quant_state(weight_quant, act_quant)
            m.set_quant_state(weight_quant, act_quant)
                
def model_bit_refactor(model, bit):
    for name, module in model.named_modules():
        if isinstance(module, (QViTBlock, QSwinBlock)):
            module.bit_refactor(bit)
        elif 'reduction' in name:
            module.bit_refactor(bit)
        

def set_8bit_headstem(model):
    # set 8-bit head-stem
    QL_list =[]
    for m in model.modules():
        if isinstance(m, (QuantLinear, QuantScalableLinear)):
            QL_list.append(m)
    
    QL_list[-1].act_quantizer.bit_candidate = [8]
    QL_list[-1].weight_quantizer.bit_candidate = [8]
    QL_list[-1].weight_quantizer.lora_type = 0
    QL_list[-1].weight_quantizer.lwc = False
    QL_list[-1].bit_refactor(8)
    QL_list[-1].is_last_layer = True

def set_quant_state_layers(layers_list,weight_quant=False, act_quant=False):
    for layer in layers_list:
        layer.set_quant_state(weight_quant, act_quant)

def layers_bit_refactor(layers_list, bit):
    for layer in layers_list:
        layer.bit_refactor(bit)

def set_quant_state_stage(stage, weight_quant=False, act_quant=False):
    for name, module in stage.named_modules():
        if isinstance(module,(QuantMatMul, QuantLinear, QuantScalableLinear)):
            module.set_quant_state(weight_quant, act_quant)

def stage_bit_refactor(stage, bit):
    for name, module in stage.named_modules():
        if isinstance(module,(QuantLinear, QuantMatMul, QuantScalableLinear)):
            module.bit_refactor(bit)

def set_bit_by_search_list(model, search_list, args):
    print(f"Set model block bitwidth by search_list: {search_list}")
    layers = model.layers if 'swin' in args.model else model.blocks
    cur_index = 0 
    while cur_index < len(layers): 
        sub_layer = layers[cur_index]
        sub_layer.bit_refactor(search_list[cur_index])
        cur_index += 1
    set_8bit_headstem(model)            