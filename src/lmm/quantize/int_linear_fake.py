import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.hadamard_utils as hadamard_utils
from quantize.quantizer import WeightQuantizer, ActQuantizer

class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        args
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_parameter('weight',org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.wbits = 16
        self.input_bits = 16
        self.output_bits = 16
        self.online_full_had=False
        self.use_temporary_parameter=False
        self.rank_multiplier= args.rank_multiplier
        self.lora_type = args.lora_type
        self.quant_type = args.quant_type
        if self.quant_type == "weight_only":
            self.Bd, self.Bu = self.make_param((self.out_features, self.in_features), 64)
        else:
            self.Bd, self.Bu = self.make_param((self.out_features, self.in_features), 48)
        
        nn.init.xavier_uniform_(self.Bu)
        
    def prepare_grouped_lora(self, bit, Xd, Xu=None):
        if self.quant_type == "weight_only":
            if bit in [2]:
                rank = 64
            elif bit in [3, 4]:
                rank = 48
            elif bit in [5, 6]:
                rank = 32
            elif bit in [7, 8]:
                rank = 16
        else:
            if bit in [4]:
                rank = 48
            elif bit in [5, 6]:
                rank = 32
            elif bit in [7, 8]:
                rank = 16
        return torch.matmul(Xd[:,:rank], Xu[:rank, :])
        
    def make_param(self, shape, rank=5):
        out_feature = shape[0]
        in_feature = shape[1]
        return nn.Parameter(torch.zeros(out_feature, rank, dtype=self.weight.dtype)), nn.Parameter(torch.zeros(rank, in_feature, dtype=self.weight.dtype))
    
    def forward(self, input: torch.Tensor):
        input_dtype = input.dtype
        # Rotate, if needed
        if self.online_full_had: #! online rotate for down_proj 
            if self.fp32_had: # Full Hadamard in FP32
                input = hadamard_utils.matmul_hadU_cuda(input.float(), self.had_K, self.K).to(input_dtype)
            else: # Full Hadamard in FP16
                input = hadamard_utils.matmul_hadU_cuda(input, self.had_K, self.K)  
                
        if self.use_temporary_parameter:
            weight = self.temp_weight
        else:
            weight = self.weight

        bias = self.bias
            
        if self.use_weight_quant and self.wbits < 16:
            weight_lora = self.prepare_grouped_lora(self.wbits, self.Bd, self.Bu)
            weight = weight + weight_lora
            weight = self.weight_quantizer(weight)

        if self.use_act_quant and self.input_bits < 16:
            input = self.input_quantizer(input)
        
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        if self.use_act_quant and self.output_bits < 16:
            out = self.output_quantizer(out)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        
    def bit_refactor(self, bit):
        if hasattr(self, 'weight_quantizer') and isinstance(self.weight_quantizer, WeightQuantizer):
             self.weight_quantizer.bit_refactor(bit)
             self.wbits = bit
        if hasattr(self, 'input_quantizer') and isinstance(self.input_quantizer, ActQuantizer):
            self.input_bits = bit
            self.input_quantizer.bit_refactor(bit) 
        if hasattr(self, 'output_quantizer') and isinstance(self.output_quantizer, ActQuantizer): ## for kv cache
            self.output_quantizer.bit_refactor(bit)

    def weight_bit_refactor(self, bit):
        if hasattr(self, 'weight_quantizer') and isinstance(self.weight_quantizer, WeightQuantizer):
             self.weight_quantizer.bit_refactor(bit)
             self.wbits = bit

    def act_bit_refactor(self, bit):
        if hasattr(self, 'input_quantizer') and isinstance(self.input_quantizer, ActQuantizer):
            self.input_bits = bit
            self.input_quantizer.bit_refactor(bit) 
    
    def prepare_for_real(self,bit):
        weight = self.weight
        weight_lora = self.prepare_path(bit, self.Bd, self.Bu)
        weight = weight + weight_lora
        
        assert hasattr(self, 'weight_quantizer') and isinstance(self.weight_quantizer, WeightQuantizer)
        scales, zeros, group_size = self.weight_quantizer.prepare_for_real_quant(weight, bit)
        return scales, zeros, group_size
    
    def prepare_for_weight(self,bit):
        weight = self.weight
        weight_lora = self.prepare_path(bit, self.Bd, self.Bu)
        weight = weight + weight_lora
        self.weight_quantizer.bit_refactor(bit)
        weight = self.weight_quantizer(weight)
        return weight
    
    def register_grad_mask(self):
        self.grad_mask_handle = []
        Bd_gradient_mask = self.Bd.new_ones(self.Bd.size())
        self.grad_mask_handle.append(self.Bd.register_hook(lambda grad: grad.mul_(Bd_gradient_mask)))
        Bu_gradient_mask = self.Bu.new_ones(self.Bu.size())
        self.grad_mask_handle.append(self.Bu.register_hook(lambda grad: grad.mul_(Bu_gradient_mask)))
        
    def update_grad_mask(self, done_bit):
        rank = 9 - done_bit
        self.grad_mask_handle = []
        Bd_gradient_mask = self.Bd.new_ones(self.Bd.size())
        Bd_gradient_mask[:,:rank*self.rank_multiplier] = False
        self.grad_mask_handle.append(self.Bd.register_hook(lambda grad: grad.mul_(Bd_gradient_mask)))
        
        Bu_gradient_mask = self.Bu.new_ones(self.Bu.size())
        Bu_gradient_mask[:rank*self.rank_multiplier, :] = False
        self.grad_mask_handle.append(self.Bu.register_hook(lambda grad: grad.mul_(Bu_gradient_mask)))
    
    def remove_grad_mask(self):
        assert len(self.grad_mask_handle) > 0
        for h in self.grad_mask_handle: 
            h.remove()