import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from .quantizer import UniformQuantizer_Weight, UniformQuantizer_Act, floor_ste

class QuantConv2d(nn.Module):
    def __init__(
        self,
        org_module: nn.Conv2d,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=True,):
        super().__init__()
        self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                dilation=org_module.dilation, groups=org_module.groups)
        self.fwd_func = F.conv2d
        self.weight = org_module.weight
        self.bias = org_module.bias
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        weight_quant_params = {}
        weight_quant_params['n_bits'] = 8
        weight_quant_params['bit_candidate'] = None
        weight_quant_params['lwc'] = False
        self.weight_quantizer = UniformQuantizer_Weight(
            **weight_quant_params,
        )
        if not disable_input_quant:
            self.act_quantizer = UniformQuantizer_Act(**act_quant_params)
        else:
            self.act_quantizer = None
            
        self.disable_input_quant = disable_input_quant
        self.ignore_reconstruction = True
        self.is_last_first_layer = True
        
    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def extra_repr(self):
        s = super().extra_repr()
        s += ", use_act_quant={}".format(self.use_act_quant)
        s += ", use_weight_quant={}".format(self.use_weight_quant)
        s += ", disable_input_quant={}".format(self.disable_input_quant)
        return s
   
class QuantLinear(nn.Module):

    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.weight = org_module.weight
        if org_module.bias is not None:
            self.bias = org_module.bias
        else:
            self.bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        
        self.weight_quantizer = UniformQuantizer_Weight(
            **weight_quant_params,
        )
        if not disable_input_quant:
            self.act_quantizer = UniformQuantizer_Act(**act_quant_params)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.ignore_reconstruction = False        
        self.is_last_layer = False
    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def extra_repr(self):
        s = super().extra_repr()
        s += ", use_act_quant={}".format(self.use_act_quant)
        s += ", use_weight_quant={}".format(self.use_weight_quant)
        s += ", disable_input_quant={}".format(self.disable_input_quant)
        return s
    
    def bit_refactor(self, bit):
        self.weight_quantizer.bit_refactor(bit)
        self.act_quantizer.bit_refactor(bit)   

class QuantScalableLinear(nn.Module):
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,

        quant_mode: str = 'vanilla' # 'vanilla', 'lora_on_scaled', 'lora_adaround'
    ):
        super().__init__()

        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.weight = org_module.weight
        self.bias = org_module.bias
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        self.rank = 48
        self.bit = 4
        self.Bd, self.Bu = self.make_param((self.out_features, self.in_features), self.rank)
        nn.init.kaiming_uniform_(tensor=self.Bu, a=math.sqrt(5))
        nn.init.zeros_(tensor=self.Bd)
        
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_input_quant = disable_input_quant
        self.ignore_reconstruction = False
        self.is_last_layer = False
        
        self.quant_mode = quant_mode
        
        self.weight_quantizer = UniformQuantizer_Weight(weight_quant_params['n_bits'], weight_quant_params['bit_candidate'], weight_quant_params['lwc'])
        if not disable_input_quant:
            self.act_quantizer = UniformQuantizer_Act(**act_quant_params)
        else:
            self.act_quantizer = None
        

    def prepare_path(self, bit, Xd, Xu):
        if bit in [4]:
            rank = 48
        elif bit in [5,6]:
            rank = 32
        elif bit in [7,8]:
            rank = 16
        return torch.matmul(Xd[:,:rank], Xu[:rank, :])
        
    def make_param(self, shape, rank=5):
        out_feature = shape[0]
        in_feature = shape[1]
        return nn.Parameter(torch.zeros(out_feature, rank)), nn.Parameter(torch.zeros(rank, in_feature))

    def forward(self, input: torch.Tensor):
        weight = self.weight
        bias = self.bias
    
        if self.use_weight_quant:

            if self.quant_mode == 'vanilla':
                weight_lora = self.prepare_path(self.bit, self.Bd, self.Bu)
                adapted_weight = weight + weight_lora
                weight = self.weight_quantizer(adapted_weight, mode='vanilla')
            
            elif self.quant_mode == 'lora_on_scaled':
                weight_lora = self.prepare_path(self.bit, self.Bd, self.Bu)
                adapted_weight = weight + weight_lora
                weight = self.weight_quantizer(adapted_weight, lora_update=weight_lora, mode='lora_on_scaled')
            
        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)
 
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def extra_repr(self):
        s = super().extra_repr()
        s += ", quant_mode={}".format(self.quant_mode)
        s += ", use_act_quant={}".format(self.use_act_quant)
        s += ", use_weight_quant={}".format(self.use_weight_quant)
        s += ", disable_input_quant={}".format(self.disable_input_quant)
        return s
    
    def bit_refactor(self, bit):
        self.weight_quantizer.bit_refactor(bit)
        if self.act_quantizer:
            self.act_quantizer.bit_refactor(bit)
        self.bit = bit
    

class QuantMatMul(nn.Module):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self,
                 act_quant_params= {}, disable_quantizer_a = False):
        super(QuantMatMul, self).__init__()
        
        
        self.quantizer_A = UniformQuantizer_Act(**act_quant_params)
        self.quantizer_B = UniformQuantizer_Act(**act_quant_params)

        self.use_act_quant = False
        self.disable_quantizer_a = disable_quantizer_a # use fp for attns. 

    def __repr__(self):
        s = super(QuantMatMul, self).__repr__()
        s = "(" + s + "MatMul_Quant={}, Disable_Quantizer_A)".format(self.use_act_quant, self.disable_quantizer_a)
        return s
    
    def set_quant_state(self, weight_quant=False, act_quant=False):
        self.use_act_quant = act_quant

    def forward(self, A, B):
        if self.use_act_quant:
            if not self.disable_quantizer_a:
                A = self.quantizer_A(A)
            B = self.quantizer_B(B)
        out = A @ B
        return out
    
    def bit_refactor(self, bit):
        self.quantizer_A.bit_refactor(bit)
        self.quantizer_B.bit_refactor(bit)