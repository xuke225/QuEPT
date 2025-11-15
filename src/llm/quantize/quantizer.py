import torch
import torch.nn as nn
import math
from utils.hadamard_utils import random_hadamard_matrix


CLIPMIN = 1e-4



def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x


def quant_activation(x, scale, bits):
    '''
    static quantization for activation with channel-wise quantization
    '''
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    scale = clamp_ste(scale,1e-4, 1e4)
    
    bs, dim1, dim2 = x.shape

    x = x.reshape(-1, dim2)
    x_int = round_ste(x / scale)
    x_int = x_int.clamp(qmin, qmax)
    x_dequant = x_int
    x_dequant = x_dequant.mul(scale)
    x_dequant = x_dequant.reshape(bs, dim1, dim2)
    return x_dequant


class WeightQuantizer(nn.Module):
    def __init__(
        self,
        n_bits,
        quantized_shape,
        group_size=[128],
        bit_list=[2],
        learnable_clipping=False,
        dtype=torch.float16
    ):
        super().__init__()
        assert 2 <= n_bits <= 32, "bitwidth not supported"
        assert len(group_size) == len(bit_list), "length of gs not equal bits, maybe there is some wrong in future"
        self.n_bits = n_bits
        self.quantized_shape = quantized_shape
        self.group_size = group_size
        self.bit_list = bit_list
        self.inc_groups = []
        self.deficiency = []
        for gs in self.group_size:
            if gs != -1:
                deficiency = self.quantized_shape[-1] % gs
                if deficiency > 0:
                    deficiency = gs - deficiency
                self.deficiency.append(deficiency)
            else:
                self.deficiency.append(-1)
        self.learnable_clipping = learnable_clipping
        if learnable_clipping:
            self.sigmoid = nn.Sigmoid()
            init_value = 4.0
            self.upbound_gs = []
            self.lowbound_gs = []
            self.upbound = []
            self.lowbound = []
            for gs in self.group_size:
                if gs != -1:
                    dims = int(quantized_shape[0] * math.ceil(quantized_shape[1] / gs))
                    upbound_factor = torch.ones((dims, 1), dtype=dtype) * init_value
                    lowbound_factor = torch.ones((dims, 1), dtype=dtype) * init_value
                    self.upbound_gs.append(upbound_factor)
                    self.lowbound_gs.append(lowbound_factor)
                else:
                    dims = self.quantized_shape[0]
                    upbound_factor = torch.ones((dims, 1), dtype=dtype) * init_value
                    lowbound_factor = torch.ones((dims, 1), dtype=dtype) * init_value
                    self.upbound.append(upbound_factor)
                    self.lowbound.append(lowbound_factor)
            if len(self.upbound_gs) > 0:
                self.upbound_gs, self.lowbound_gs = nn.Parameter(torch.stack(self.upbound_gs)), nn.Parameter(torch.stack(self.lowbound_gs))
            if len(self.upbound) > 0:
                self.upbound, self.lowbound = nn.Parameter(torch.stack(self.upbound)), nn.Parameter(torch.stack(self.lowbound))

    def forward(self, x: torch.Tensor, lora_update: torch.Tensor = None, mode: str = 'vanilla'):
        # ================== Vanilla (W+BA) -> Quantize ==================
        if mode == 'vanilla':
            if lora_update is not None:
                x = x + lora_update
        elif mode not in ['lora_on_scaled']:
            raise ValueError(f"Unsupported mode: {mode}. Must be 'vanilla' or 'lora_on_scaled'.")
        # =======================================================================
        
        if self.n_bits >= 16:
            return x

        bit_index = self.bit_list.index(self.n_bits)
        gs = self.group_size[bit_index]
        
        original_shape = x.shape
        x_reshaped = x

        if gs != -1 and gs != original_shape[-1]:
            current_deficiency = self.deficiency[bit_index]
            if current_deficiency > 0:
                pad_zeros = torch.zeros((x.shape[0], current_deficiency), dtype=x.dtype, device=x.device)
                x_reshaped = torch.cat((x, pad_zeros), dim=1)
            
            x_reshaped = x_reshaped.reshape(-1, gs)

        xmin = x_reshaped.amin([-1], keepdim=True)
        xmax = x_reshaped.amax([-1], keepdim=True)

        if self.learnable_clipping:
            if bit_index < len(self.upbound_gs):
                upbound_factor = self.sigmoid(self.upbound_gs[bit_index])
                lowbound_factor = self.sigmoid(self.lowbound_gs[bit_index])
            else:
                bound_index = bit_index - len(self.upbound_gs)
                upbound_factor = self.sigmoid(self.upbound[bound_index])
                lowbound_factor = self.sigmoid(self.lowbound[bound_index])
            
            xmin = xmin * lowbound_factor
            xmax = xmax * upbound_factor
            
        quant_range = xmax - xmin
        qmin = 0
        qmax = 2 ** self.n_bits - 1
        scale = quant_range / qmax
        scale = clamp_ste(scale, min=1e-5, max=1e4)
        round_zero_point = -(xmin / scale).clamp(min=-1e4, max=1e4).round()

        x_scaled = x_reshaped / scale
        
        # ================== (W/s + BA/s) -> Round ==================
        if mode == 'lora_on_scaled':
            if lora_update is not None:
                lora_update_reshaped = lora_update
                if gs != -1 and gs != original_shape[-1]:
                    current_deficiency = self.deficiency[bit_index]
                    if current_deficiency > 0:
                        pad_zeros = torch.zeros((lora_update.shape[0], current_deficiency), dtype=lora_update.dtype, device=lora_update.device)
                        lora_update_reshaped = torch.cat((lora_update, pad_zeros), dim=1)
                    lora_update_reshaped = lora_update_reshaped.reshape(-1, gs)
                
                x_scaled = x_scaled + (lora_update_reshaped / scale)
            else:
                import warnings
                warnings.warn("'lora_on_scaled' mode is active but lora_update is None.")
        # =======================================================================
        
        x_int = round_ste(x_scaled)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        
        x_int = x_int.clamp(qmin, qmax)
        x_dequant = x_int
        
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        
        x_dequant = x_dequant.mul(scale)

        if gs != -1 and gs != original_shape[-1]:
            padded_shape_1 = original_shape[0]
            padded_shape_2 = original_shape[1] + self.deficiency[bit_index]
            x_dequant = x_dequant.reshape(padded_shape_1, padded_shape_2)
            
            if self.deficiency[bit_index] > 0:
                x_dequant = x_dequant[:, :original_shape[1]]

        return x_dequant

    def bit_refactor(self, bit):
        self.n_bits = bit
    
class ActQuantizer(nn.Module):
    def __init__(
        self,
        n_bits,
        bit_list = [2],
        
    ):
        super().__init__()
        assert 2 <= n_bits <= 32, "bitwidth not supported"
        self.n_bits = n_bits
        self.bit_list = bit_list     
    def forward(self, x:torch.Tensor):
        if self.n_bits >= 16:
            return x

        xmin = x.amin([-1], keepdim=True)
        xmax =  x.amax([-1], keepdim=True)
        xmin = xmin 
        xmax = xmax 
        abs_max = torch.max(xmax.abs(),xmin.abs())
        scale = abs_max / (2**(self.n_bits-1)-1)
        scale_1 = 1 / scale
        scale = clamp_ste(scale, min=1e-4, max=1e4)
        x_int = round_ste(x * scale_1)
        x_dequant = x_int
        x_dequant = x_dequant.mul(scale)
        return x_dequant
          
    
    def bit_refactor(self, bit):
        self.n_bits = bit

class KVQuantizer(nn.Module):
    def __init__(
        self,
        n_bits,
        quantized_shape,
        group_size=128,
    ):
        super().__init__()
        assert 2 <= n_bits <= 32, "bitwidth not supported"
        self.n_bits = n_bits
        self.quantized_shape = quantized_shape
        self.group_size = group_size
        assert quantized_shape[-1] % self.group_size == 0 
        self.inc_groups = quantized_shape[-1] // self.group_size
        self.clip_ratio = 0.95
    def forward(self, x:torch.Tensor):
        if self.n_bits >= 16:
            return x

        init_shape = x.shape
        reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // self.group_size, self.group_size)
        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1
        qmin = 0
        qmax = 2 ** (self.n_bits) - 1
        scale = (xmax - xmin) / (qmax)
        zero = round_ste(-xmin / scale)
        scale = scale.repeat(1, 1, 1, self.group_size).reshape(init_shape)
        zero = zero.repeat(1, 1, 1, self.group_size).reshape(init_shape)
        x_int = clamp_ste(round_ste(x / scale) + zero, qmin, qmax)
        x_dequant = scale * (x_int - zero)
        return x_dequant
    
   
    
    def bit_refactor(self, bit):
        self.n_bits = bit