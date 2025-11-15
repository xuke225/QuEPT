import torch

# @torch.no_grad()
# def pseudo_quantize_tensor(tensor, n_bits=8, zero_point=True, q_group_size=-1, per_tensor=False, inplace=False):
#     """
#     The basic quantization function for weight, activation and KV cache.
#     """
#     org_tensor_shape = tensor.shape
#     if q_group_size > 0:
#         assert org_tensor_shape[-1] % q_group_size == 0
#         tensor = tensor.reshape(-1, q_group_size)
#     if per_tensor:
#         tensor = tensor.reshape(1, -1)
#     assert tensor.dim() == 2
#     if zero_point:
#         max_val = tensor.amax(dim=1, keepdim=True)
#         min_val = tensor.amin(dim=1, keepdim=True)
#         max_int = 2**n_bits - 1
#         min_int = 0
#         scales = (max_val - min_val).clamp(min=1e-5) / max_int
#         zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
#     else:
#         max_val = tensor.abs().amax(dim=1, keepdim=True)
#         max_val = max_val.clamp(min=1e-5)
#         max_int = 2 ** (n_bits - 1) - 1
#         min_int = -(2 ** (n_bits - 1))
#         scales = max_val / max_int
#         zeros = 0

#     if inplace:
#         (
#             (tensor.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
#         ).mul_(scales)
#     else:
#         tensor = (
#             torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int) - zeros
#         ) * scales

#     assert torch.isnan(tensor).sum() == 0

#     tensor = tensor.reshape(org_tensor_shape)

#     # return the quantized tonsor, the scaling factor and the zero point value
#     # return tensor, scales.view(tensor.shape[0], -1), zeros.view(tensor.shape[0], -1)
#     return tensor

@torch.no_grad()
def pseudo_quantize_tensor(tensor, n_bits=8, zero_point=True, q_group_size=-1, per_tensor=False, inplace=False):
    org_tensor_shape = tensor.shape
    if q_group_size > 0:
        dim_size = org_tensor_shape[-1]
        remainder = dim_size % q_group_size
        if remainder != 0:
            groups = []
            group_start = 0
            while group_start < dim_size:
                group_end = min(group_start + q_group_size, dim_size)
                groups.append((group_start, group_end))
                group_start = group_end
            
            processed_groups = []
            for start, end in groups:
                group = tensor[..., start:end]
                group_size = end - start
                group_2d = group.reshape(-1, group_size)
                quantized_group_2d = pseudo_quantize_tensor(
                    group_2d, n_bits, zero_point, q_group_size=group_size, 
                    per_tensor=per_tensor, inplace=inplace
                )
                quantized_group = quantized_group_2d.reshape(group.shape)
                processed_groups.append(quantized_group)
            
            tensor = torch.cat(processed_groups, dim=-1).reshape(org_tensor_shape)
            return tensor
        else:
            tensor = tensor.reshape(-1, q_group_size)
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2
    if zero_point:
        max_val = tensor.amax(dim=1, keepdim=True)
        min_val = tensor.amin(dim=1, keepdim=True)
        max_int = 2**n_bits - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:
        max_val = tensor.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        max_int = 2**(n_bits - 1) - 1
        min_int = -2**(n_bits - 1)
        scales = max_val / max_int
        zeros = 0
    if inplace:
        tensor.div_(scales).round_().add_(zeros).clamp_(min_int, max_int).sub_(zeros).mul_(scales)
    else:
        tensor = (torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int) - zeros) * scales
    tensor = tensor.reshape(org_tensor_shape)
    return tensor

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8, zero_point=False):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    tensor = pseudo_quantize_tensor(w, n_bits=n_bits, zero_point=zero_point, q_group_size=-1, per_tensor=False, inplace=False)
    return tensor
    
@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8, zero_point=False):
    t_shape = t.shape
    t = t.view(-1, t_shape[-1])
    t = pseudo_quantize_tensor(t, n_bits=n_bits, zero_point=zero_point, q_group_size=-1, per_tensor=False, inplace=False)
    return t.reshape(t_shape)
    
@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8, zero_point=False):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    tensor = pseudo_quantize_tensor(w, n_bits=n_bits, zero_point=zero_point, q_group_size=-1, per_tensor=True, inplace=False)
    return tensor
    
@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8, zero_point=False):
    t_shape = t.shape
    t = t.view(-1, t_shape[-1])
    t = pseudo_quantize_tensor(t, n_bits=n_bits, zero_point=zero_point, q_group_size=-1, per_tensor=True, inplace=False)
    return t.reshape(t_shape)