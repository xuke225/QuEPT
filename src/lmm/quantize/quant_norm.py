import torch
import torch.nn as nn
from quantize.quantizer import ActQuantizer

class QuantRMSNorm(nn.Module):
    def __init__(self, 
                ori_norm,
                output_bits=16,
                args = None
                ):
        super().__init__()
        self.register_buffer('weight',ori_norm.weight)
        self.bias = None
        self.variance_epsilon = ori_norm.variance_epsilon
        self.use_temporary_parameter = False
        self.use_act_quant = False
        self.output_bits = output_bits
        self.output_quantizer = ActQuantizer(args.input_bits, args.input_bit_list)
        
    def forward(self, x):
        if self.use_temporary_parameter:
            weight = self.temp_weight
        else:
            weight = self.weight

        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x =  x.to(input_dtype) * weight

        if self.use_act_quant and self.output_bits < 16:
            x = self.output_quantizer(x)
            
        return x

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
    
    def bit_refactor(self, bit):
        self.output_bits = bit
        self.output_quantizer.bit_refactor(bit)