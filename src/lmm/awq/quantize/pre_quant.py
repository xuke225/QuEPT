import torch
import torch.nn as nn
import tqdm
import gc
import functools
from collections import defaultdict

from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from .auto_scale import auto_scale_block, apply_scale
from .auto_clip import auto_clip_block, apply_clip, auto_clip_block_multi_bit, apply_clip_multi_bit
from .auto_scale import auto_scale_block_vlm

from utils.search import append_str_prefix, get_op_name
from ..utils.calib_data import get_vlm_calib_dataset
from calibration.coco_vl import get_multimodal_calib_examples


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    if model.__class__.__name__ == 'LlamaForCausalLM':
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaQwenForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "InternLM2ForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "InternVLChatModel":
        layers = model.language_model.model.layers
    elif model.__class__.__name__ == "InternVL2":
        layers = model.language_model.model.layers
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    else: 
        raise NotImplementedError(type(model)) 
    return layers

def get_vision_blocks(model):
    if model.__class__.__name__ == "LlavaLlamaForCausalLM":
        layers = model.model.vision_tower.vision_tower.vision_model.encoder.layers
    elif model.__class__.__name__ == "LlavaQwenForCausalLM":
        layers = model.model.vision_tower.vision_tower.vision_model.encoder.layers
    elif model.__class__.__name__ == "InternVLChatModel":
        layers = model.vision_model.encoder.layers
    elif model.__class__.__name__ == "InternVL2":
        layers = model.vision_model.encoder.layers
    else: 
        raise NotImplementedError(type(model)) 
    return layers
    
def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(device)
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    elif "bigcode" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)
    elif "neox" in str(model.__class__).lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device)
        model.embed_out = model.embed_out.to(device)
    elif model.__class__.__name__ == "LlavaQwenForCausalLM":
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        # model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif model.__class__.__name__ == "InternLM2ForCausalLM":
        model.tok_embeddings = model.tok_embeddings.to(device)
    elif "InternVL" in model.__class__.__name__:
        if  hasattr(model.language_model, 'model') and hasattr(model.language_model.model,'tok_embeddings'):
            model.language_model.model.tok_embeddings = model.language_model.model.tok_embeddings.to(device)
        else:
            model.language_model.model.embed_tokens = model.language_model.model.embed_tokens.to(device)
    elif model.__class__.__name__ == "Qwen2VLForConditionalGeneration":
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    else:
        raise NotImplementedError(type(model))
    
def move_vision_embed(model, device):
    if model.__class__.__name__ == "LlavaQwenForCausalLM":
        model.model.vision_tower.vision_tower.vision_model.embeddings = model.model.vision_tower.vision_tower.vision_model.embeddings.to(device)
        # model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif model.__class__.__name__ == "Qwen2VLForConditionalGeneration":
        model.model.vision_tower.vision_tower.vision_model.embeddings = model.model.vision_tower.vision_tower.vision_model.embeddings.to(device)
    elif model.__class__.__name__ == "InternVLChatModel":
        model.vision_model.embeddings = model.vision_model.embeddings.to(device)
    elif model.__class__.__name__ == "InternVL2":
        model.vision_model.embeddings = model.vision_model.embeddings.to(device)
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        model.model.vision_tower.vision_tower.vision_model.embeddings = model.model.vision_tower.vision_tower.vision_model.embeddings.to(device)
    else:
        raise NotImplementedError(type(model))
    
def process_input(prompt_inputs, prompt_kwargs):
    inputs = {**prompt_inputs, **prompt_kwargs}
    inputs["use_cache"] = False
    vision_mask = inputs.pop("vision_mask", None)
    caption_mask = inputs.pop("caption_mask", None)
    
    return inputs, vision_mask, caption_mask

@torch.no_grad()
def run_awq_llm(
    model, enc,
    w_bit, q_config, args,
    n_samples=128, seqlen=512,
    auto_scale=True, mse_range=True,
    calib_data="pileval",
):
    from ..utils.calib_data import get_lm_calib_dataset
    from ..utils.module import append_str_prefix, get_op_name


    layers = get_blocks(model)

    samples = get_lm_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, seq_len=seqlen)
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")
    
    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")
    
    gc.collect()
    torch.cuda.empty_cache()

    awq_results = {
        "scale": [],
        "clip": [],
    }

    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name,
                                  feat_dict=input_feat)))
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()

        if auto_scale:  # if it applies, we should also modify the input_feat with scales
            scales_list = auto_scale_block(
                layer, layer_kwargs,
                w_bit=w_bit, q_config=q_config,
                input_feat=input_feat,
            )
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            # append prefix to make names global
            awq_results["scale"] += append_str_prefix(scales_list, get_op_name(model, layer) + ".")

        # Clear GPU memory
        torch.cuda.empty_cache()
        
        if mse_range:
            clip_list = auto_clip_block_multi_bit(layer, args=args,
                            input_feat=input_feat,)
            # apply_clip(layer, clip_list)
            # append prefix to make names global
            awq_results["clip"] += append_str_prefix(clip_list, get_op_name(model, layer) + ".")

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()
        
    return awq_results

@torch.no_grad()
def run_awq_vlm(
    args,
    model,
    lm,
    w_bit,
    q_config,
    auto_scale=True,
    mse_range=True,
    calib_data="coco",
    loss_mode="mae"
):

    examples = get_multimodal_calib_examples(data_path=args.data_path,
        image_folder=args.image_folder, model=model,
        n_samples=args.train_size, few_shot_format=args.few_shot_format,
        interleave_format=args.interleave_format, text_data_path=args.text_data_path,
        shuffle=True
)
    prompt_inputs, prompt_kwargs = model.generate_input(examples)
    if "bigcode" in str(model.__class__).lower():
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to("cuda")

    ans_mask = None
    vis_mask = None

    if args.quant_vision:
        vis_layers = get_vision_blocks(model.model)

    layers = get_blocks(model.model)

    inps = []
    vis_inps = []
    layer_kwargs = {}
    vis_layer_kwargs = {}
    attn_mask = None
    causal_mask = None

    if args.quant_vision:
        vis_layers[0] = vis_layers[0].cuda()
        move_vision_embed(model.model, "cuda")

    layers[0] = layers[0].cuda()
    move_embed(model.model, "cuda")

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference
    
    class Vis_Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        # def forward(self, inps, attn_mask, **kwargs): # for llavaov
        def forward(self, inps, attn_mask, causal_mask, **kwargs): # for llava
            vis_inps.append(inps)
            attn_mask = attn_mask # for llavaov, llava
            causal_mask = causal_mask # for llava
            vis_layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    if args.quant_vision:
        vis_layers[0] = Vis_Catcher(vis_layers[0])
    layers[0] = Catcher(layers[0])
    inputs, vision_mask, caption_mask = process_input(prompt_inputs, prompt_kwargs)

    model.to_cuda()
    try:
        prompt_inputs, prompt_kwargs = model.generate_input(examples) # Vision Encoder input
    except ValueError: # work with early exit
        pass

    if args.quant_vision:
        vis_layers[0] = vis_layers[0].module  # restore
        vis_inps = vis_inps[0]

        vis_layers[0] = vis_layers[0].cpu()
        move_vision_embed(model.model, "cpu")

    try:
        model(**inputs) # VLM input
    except ValueError: # work with early exit
        pass

    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model.model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    awq_results = {
        "scale": [],
        "clip": [],
    }

    # for i in tqdm.tqdm(range(len(vis_layers)), desc="Running AWQ For Vision Encoder..."):
    #     vis_layer = vis_layers[i]
    #     vis_layer = vis_layer.cuda()
    #     named_linears = get_named_linears(vis_layer)

    #     scale_reweight_ratio_dict = {
    #             "attn": None,
    #             "mlp": None
    #     }

    #     # firstly, get input features of all linear layers
    #     def cache_input_hook(m, x, y, name, feat_dict):
    #         x = x[0]
    #         x = x.detach().cpu()
    #         feat_dict[name].append(x)

    #     input_feat = defaultdict(list)
    #     handles = []
    #     for name in named_linears:
    #         handles.append(
    #             named_linears[name].register_forward_hook(
    #                 functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
    #             )
    #         )
    #     vis_inps = vis_inps.to(next(vis_layer.parameters()).device)  # in case multi-gpu
    #     # get output as next layer's input
    #     # vis_inps = vis_layer(vis_inps, attn_mask, **vis_layer_kwargs)[0] # for llavaov
    #     vis_inps = vis_layer(vis_inps, attn_mask, causal_mask, **vis_layer_kwargs)[0] # for llava
    #     for h in handles:
    #         h.remove()
    #     # now solve for scaling
    #     input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

    #     # Clear GPU memory
    #     torch.cuda.empty_cache()

    #     if (
    #         auto_scale
    #     ):  # if it applies, we should also modify the input_feat with scales
    #         scales_list = auto_scale_block_vlm(
    #             vis_layer,
    #             vis_layer_kwargs,
    #             w_bit=w_bit,
    #             q_config=q_config,
    #             input_feat=input_feat,
    #             ans_mask=ans_mask,
    #             vis_mask=vis_mask,
    #             reweight_ratio_dict=scale_reweight_ratio_dict,
    #             loss_mode=loss_mode
    #         )
    #         # apply_scale(layer, scales_list, input_feat_dict=input_feat)
    #         apply_scale(vis_layers[i], scales_list, input_feat_dict=input_feat)
    #         # append prefix to make names global
    #         awq_results["scale"] += append_str_prefix(
    #             scales_list, get_op_name(model.model, vis_layer) + "."
    #         )

    #     if mse_range:
    #         clip_list = auto_clip_block_multi_bit(vis_layer, args=args,
    #                         input_feat=input_feat,)
    #         # apply_clip(layer, clip_list)
    #         # append prefix to make names global
    #         awq_results["clip"] += append_str_prefix(
    #             clip_list, get_op_name(model.model, vis_layer) + "."
    #         )

    #     # Clear GPU memory
    #     torch.cuda.empty_cache()

    #     vis_layer = vis_layer.cpu()
    #     # Haotian: check activation replacement
    #     del input_feat

    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ For VLM..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        scale_reweight_ratio_dict = {
                "attn": None,
                "mlp": None
        }

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()

        if (
            auto_scale
        ):  # if it applies, we should also modify the input_feat with scales
            scales_list = auto_scale_block_vlm(
                layer,
                layer_kwargs,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
                ans_mask=ans_mask,
                vis_mask=vis_mask,
                reweight_ratio_dict=scale_reweight_ratio_dict,
                loss_mode=loss_mode
            )
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            # append prefix to make names global
            awq_results["scale"] += append_str_prefix(
                scales_list, get_op_name(model.model, layer) + "."
            )

        if mse_range:
            clip_list = auto_clip_block_multi_bit(layer, args=args,
                            input_feat=input_feat,)
            # apply_clip(layer, clip_list)
            # append prefix to make names global
            awq_results["clip"] += append_str_prefix(
                clip_list, get_op_name(model.model, layer) + "."
            )

        # Clear GPU memory
        torch.cuda.empty_cache()

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    return awq_results

def apply_awq(model, awq_results):
    apply_scale(model, awq_results["scale"])
    apply_clip(model, awq_results["clip"])
