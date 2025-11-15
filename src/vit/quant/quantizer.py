import math
import numpy as np
import torch
from tqdm import *
import torch.nn as nn
import torch.nn.functional as F

CLIPMIN = 1e-5

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def ceil_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for celi operation.
    """
    return (torch.ceil(x)-x).detach() + x

def floor_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for floor operation.
    """
    return (torch.floor(x)-x).detach() + x

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()

class UniformQuantizer_Weight(nn.Module):
    def __init__(self, n_bits: int = 8, bit_candidate: list = None, lwc = False):
        """
            weight quantizer is channel_wise as default
            lwc means learnable weight clip 
        """
        super(UniformQuantizer_Weight, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta_list = None
        self.zero_point_list = None
        self.inited = False
        self.lwc = lwc
        
        #multi-bit 
        self.bit_candidate = [n_bits] if bit_candidate is None else bit_candidate
        
        
    def __repr__(self):
        s = super(UniformQuantizer_Weight, self).__repr__()
        s = "(" + s + " inited={}, cur_bit={}, bit_candidate={}, lwc={})".format(self.inited, self.n_bits, self.bit_candidate, self.lwc)
        return s

    def init_lwc(self, x:torch.Tensor):
        init_value = 4.
        dim1 = x.shape[0]
        upbound = []
        lowbound = []
        for b in self.bit_candidate:
            temp_upbound = x.new_ones((dim1,1))
            temp_lowbound = x.new_ones((dim1, 1))
            upbound.append(temp_upbound*init_value)
            lowbound.append(temp_lowbound*init_value)
        
        return nn.Parameter(torch.stack(upbound)), nn.Parameter(torch.stack(lowbound))

    def forward(self, x: torch.Tensor, lora_update: torch.Tensor = None, mode: str = 'vanilla'):
        if self.inited is False:
            if self.lwc:
                self.upbound_factor, self.lowbound_factor = self.init_lwc(x)
            self.inited = True

        # ================== Vanilla (W+BA) -> Quantize ==================
        if mode == 'vanilla':
            pass
        # =======================================================================
        
        if self.lwc:
            bit_index = self.bit_candidate.index(self.n_bits)
            reduce_shape = [-1]
            xmin = x.amin(reduce_shape, keepdim=True)
            xmax = x.amax(reduce_shape, keepdim=True)
            clip_max = F.sigmoid(self.upbound_factor[bit_index]) * xmax
            clip_min = F.sigmoid(self.lowbound_factor[bit_index]) * xmin
            range_val = clip_max - clip_min
            scale = range_val / (self.n_levels - 1)
            scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (-clip_min / scale).round()
        else:
            reduce_shape = [-1]
            xmin = x.amin(reduce_shape, keepdim=True)
            xmax = x.amax(reduce_shape, keepdim=True)
            range_val = xmax - xmin
            scale = range_val / (self.n_levels - 1)
            scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (-xmin / scale).round()

        x_scaled = x / scale
        
        # ================== (W/s + BA) -> Round ==================
        if mode == 'lora_on_scaled':
            if lora_update is not None:
                x_scaled = x_scaled + lora_update
            else:
                import warnings
                warnings.warn("'lora_on_scaled' mode is active but lora_update is None.")
        # ===============================================================

        x_int = round_ste(x_scaled) + zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - zero_point) * scale
        
        return x_dequant

    def bit_refactor(self, bit:int):
        assert bit in self.bit_candidate, 'bitwidth not supported'
        self.n_bits = bit
        self.n_levels = 2 ** bit



class UniformQuantizer_Act(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, bit_candidate: list = None, prob: float = 1.0):
        super(UniformQuantizer_Act, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta_list = None
        self.zero_point_list = None
        self.inited = False
        self.channel_wise = channel_wise
        
        #multi-bit 
        self.bit_candidate = [n_bits] if bit_candidate is None else bit_candidate
        
        '''do like dropout'''
        self.prob = prob
        self.is_training = False
        
    def __repr__(self):
        s = super(UniformQuantizer_Act, self).__repr__()
        s = "(" + s + " inited={}, channel_wise={}, cur_bit={}, bit_candidate={})".format(self.inited, self.channel_wise, self.n_bits, self.bit_candidate)
        return s
    
    def forward(self, x: torch.Tensor):
        if self.inited is False:
            self.delta_list, self.zero_point_list = self.init_quantization_scale_multi_bit(x, self.channel_wise)
            self.inited = True

        # start quantization
        bit_index = self.bit_candidate.index(self.n_bits)
        # activations per-tensor: activations for the scale is inited by percentile observers and fine-tuned in block recstruction.
        x_int = round_ste(x / self.delta_list[bit_index]) 
        x_quant = torch.clamp(x_int + self.zero_point_list[bit_index], 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point_list[bit_index]) * self.delta_list[bit_index]

        if self.is_training and self.prob < 1.0:
            x_ans = torch.where(torch.rand_like(x) < self.prob,x_dequant, x)
        else:
            x_ans = x_dequant
        return x_ans

    def init_quantization_scale_multi_bit(self,x: torch.Tensor, channel_wise: bool = False):
        detla_list, zero_point_list = [], []
        
        for b in self.bit_candidate:
            temp_detla, temp_zero_point = self.init_quantization_scale(x, channel_wise, b)
            detla_list.append(temp_detla)
            zero_point_list.append(temp_zero_point)
        
        return torch.stack(detla_list), torch.stack(zero_point_list)
    
    def init_quantization_scale(self, x: torch.Tensor, channel_wise, bit: int):
        
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=-1)[0]
            elif len(x.shape) == 3:     
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False, bit=bit)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False, bit=bit)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()

            best_score = 1e+10
            search_range = [0.999, 0.9999, 0.99999]
            for pct in search_range:
                try:
                    new_max = torch.quantile(x_clone.reshape(-1), pct)
                    new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                except:
                    new_max = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), pct * 100),
                        device=x_clone.device,
                        dtype=torch.float32)
                    new_min = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                        device=x_clone.device,
                        dtype=torch.float32)   
                x_q = self.quantize(x_clone, new_max, new_min, bit)
               
                score = lp_loss(x_clone, x_q, p=2.0, reduction='all')
                
                if score < best_score:
                    best_score = score
                    delta = (new_max - new_min) / (2 ** bit - 1)
                    delta = delta.clamp(min=CLIPMIN, max=1e4)
                    zero_point = (- new_min / delta).round()
                    
        return delta, zero_point

    def quantize(self, x, max, min, bit):
        delta = (max - min) / (2 ** bit - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = round_ste(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0,  (2 ** bit) - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bit_refactor(self, bit:int):
        assert bit in self.bit_candidate, 'bitwidth not supported'
        self.n_bits = bit
        self.n_levels = 2 ** bit

