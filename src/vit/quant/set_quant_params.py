import torch
from .quant_modules import QuantLinear, QuantMatMul, QuantConv2d, QuantScalableLinear
from typing import Union

def set_weight_quantize_params(model):
    for module in model.modules():
        if isinstance(module, (QuantLinear, QuantConv2d, QuantScalableLinear)):
            module.weight_quantizer.inited = False
            module.weight_quantizer(module.weight)
            
def save_quantized_weight(model):
    for module in model.modules():
        if isinstance(module, QuantLinear, QuantScalableLinear):
            module.weight.data = module.weight_quantizer(module.weight)

def set_act_quantize_params(module, cali_data, weight_quant, act_quant, batch_size: int = 128):
    
    if hasattr(module,'set_quant_state'):
        module.set_quant_state(weight_quant, act_quant)
    else:
        for m in module.named_modules():
            if hasattr(m,'set_quant_state'):
                m.set_quant_state(weight_quant, act_quant)

    for t in module.modules():
        if isinstance(t, (QuantLinear, QuantScalableLinear)):
            t.act_quantizer.inited = False
        elif isinstance(t, QuantMatMul):
            t.quantizer_A.inited = False
            t.quantizer_B.inited = False
    '''set or init step size and zero point in the activation quantizer'''
    batch_size = min(batch_size, cali_data.size(0))
    with torch.no_grad():
        for i in range(int(cali_data.size(0) / batch_size)):
            module(cali_data[i * batch_size:(i + 1) * batch_size].cuda())

    # for i in range(int(cali_data.size(0) / batch_size)):
    #     module(cali_data[i * batch_size:(i + 1) * batch_size].cuda())
    torch.cuda.empty_cache()

