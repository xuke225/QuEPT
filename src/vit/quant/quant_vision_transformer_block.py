import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from .quant_modules import QuantLinear, QuantMatMul, QuantScalableLinear
from timm.models.vision_transformer import Attention, Block
from timm.models.layers import Mlp
from timm.models.swin_transformer import SwinTransformerBlock, WindowAttention, window_partition,window_reverse
from copy import deepcopy
from typing import Optional, Union

class QViTMLP(nn.Module):
    def __init__(
        self,
        org_module: Union[nn.Module, Mlp],
        args=None,
    ):
        super().__init__()
        fc1_quant_params = deepcopy(args.act_quant_params)
        fc1_quant_params['channel_wise'] = True
        if args.scaleLinear :
            self.fc1 = QuantScalableLinear(org_module.fc1, args.weight_quant_params, fc1_quant_params)
        else:
            self.fc1 = QuantLinear(org_module.fc1, args.weight_quant_params,fc1_quant_params)
        self.act = org_module.act
        if args.scaleLinear :
            self.fc2 = QuantScalableLinear(org_module.fc2, args.weight_quant_params, args.act_quant_params)
        else:
            self.fc2 = QuantLinear(org_module.fc2, args.weight_quant_params,args.act_quant_params)
        self.drop = org_module.drop
        self.is_training = False
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class QViTAttention(nn.Module):
    def __init__(self, org_module: Union[nn.Module, Attention], args=None,):
        super().__init__()
        self.num_heads = org_module.num_heads
        # self.head_dim = org_module.head_dim
        self.scale = org_module.scale
        self.save_attention = False
        self.attn_values = None
        self.matmul1 = QuantMatMul(args.act_quant_params)
        self.matmul2 = QuantMatMul(args.act_quant_params, disable_quantizer_a = True)

        qkv_quant_params = deepcopy(args.act_quant_params)
        qkv_quant_params['channel_wise'] = True
        if args.scaleLinear :
            self.qkv = QuantScalableLinear(org_module.qkv, args.weight_quant_params, qkv_quant_params)
        else:
            self.qkv = QuantLinear(org_module.qkv, args.weight_quant_params, qkv_quant_params)

        self.attn_drop = org_module.attn_drop
        if args.scaleLinear :
            self.proj = QuantScalableLinear(org_module.proj, args.weight_quant_params, args.act_quant_params)
        else:
            self.proj = QuantLinear(org_module.proj, args.weight_quant_params, args.act_quant_params)
        self.proj_drop = org_module.proj_drop


    def forward(self, x):
        B, N, C = x.shape
        # import pdb 
        # pdb.set_trace()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.save_attention:
            self.attn_values = self.matmul1(q, k.transpose(-2, -1))
            # self.attn_values = F.normalize(attn_values, p=2, dim=-1)
        attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
         
            
        attn = self.attn_drop(attn)
        del q, k

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
        del attn, v
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class QViTBlock(nn.Module):
    def __init__(self, org_module: Union[nn.Module, Block], args=None,):
        super().__init__()
        self.norm1 = org_module.norm1
        self.attn = QViTAttention(org_module=org_module.attn,args=args)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        
        self.drop_path = org_module.drop_path
        self.norm2 = org_module.norm2
        self.mlp = QViTMLP(org_module=org_module.mlp, args=args)
        # outlier handle flags
        self.outlier_handle_flag = False
        self.is_training = False
    def forward(self, x):
        if self.is_training:
            norm1_out = self.norm1(x)
            mhsa_out = self.drop_path(self.attn(norm1_out))
            x = x + mhsa_out
            norm2_out = self.norm2(x)
            ffn_out = self.drop_path(self.mlp(norm2_out))
            x = x + ffn_out
            return x, norm1_out, mhsa_out, norm2_out, ffn_out
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

 
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul, QuantScalableLinear)):
                m.set_quant_state(weight_quant, act_quant) 
                
    def bit_refactor(self, bit):
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul, QuantScalableLinear)):
                m.bit_refactor(bit)
                
    def random_bit_refactor(self, bit_choices = [4,5,6,7,8], random_type = 'L'):
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul, QuantScalableLinear)):
                if random_type == 'H':
                    m.bit_refactor(bit_choices[-1])
                elif random_type == 'L':
                    m.bit_refactor(bit_choices[0])
                elif random_type == 'R':
                    m.bit_refactor(random.choice(bit_choices))
        
class QSwinBlock(nn.Module):

    def __init__(self, org_module: Union[nn.Module, SwinTransformerBlock], args=None,):
        super().__init__()
        self.dim = org_module.dim
        self.input_resolution = org_module.input_resolution
        self.num_heads = org_module.num_heads
        self.window_size = org_module.window_size
        self.shift_size = org_module.shift_size
        self.mlp_ratio = org_module.mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = org_module.norm1
        self.attn = QWindowAttention(org_module.attn, args)
        self.drop_path = org_module.drop_path
        self.norm2 = org_module.norm2
        self.mlp = QViTMLP(org_module=org_module.mlp, args=args)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
    
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        names = []
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul, QuantScalableLinear)):
                names.append(name)
                m.set_quant_state(weight_quant, act_quant) 
                
    def bit_refactor(self, bit):
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul, QuantScalableLinear)):
                m.bit_refactor(bit)

class QWindowAttention(nn.Module):
    def __init__(self, org_module: Union[nn.Module, WindowAttention], args=None,):
        super().__init__()

        self.dim = org_module.dim
        self.window_size = org_module.window_size
        self.num_heads = org_module.num_heads
        self.scale = org_module.scale
        
        self.relative_position_bias_table = org_module.relative_position_bias_table
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        self.softmax =  org_module.softmax

        self.matmul1 = QuantMatMul(args.act_quant_params)
        self.matmul2 = QuantMatMul(args.act_quant_params, disable_quantizer_a = True)

        qkv_quant_params = deepcopy(args.act_quant_params)
        qkv_quant_params['channel_wise'] = True
        if args.scaleLinear:
            self.qkv = QuantScalableLinear(org_module.qkv, args.weight_quant_params, qkv_quant_params)
        else:
            self.qkv = QuantLinear(org_module.qkv, args.weight_quant_params, qkv_quant_params)
        self.attn_drop = org_module.attn_drop
        if args.scaleLinear:
            self.proj = QuantScalableLinear(org_module.proj, args.weight_quant_params, args.act_quant_params)
        else:
            self.proj = QuantLinear(org_module.proj, args.weight_quant_params, args.act_quant_params)
        self.proj_drop = org_module.proj_drop
        
        self.save_attention = False
        self.attn_values = None

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        # attn = (q @ k.transpose(-2, -1))
        if self.save_attention:
            self.attn_values = self.matmul1(q, k.transpose(-2, -1))
        attn = self.matmul1(q, k.transpose(-2,-1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        
        
    