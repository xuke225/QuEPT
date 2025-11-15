# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.
import abc
import torch
import typing
import sys
import json
import hashlib
import transformers
import utils
import os
import logging
import collections
import torch.nn.functional as F
from typing import Iterable
from tqdm import tqdm
from transformers.cache_utils import DynamicCache
from abc import abstractmethod
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

OPT_MODEL = transformers.models.opt.modeling_opt.OPTForCausalLM
OPT_LAYER = transformers.models.opt.modeling_opt.OPTDecoderLayer
OPT_NORM = torch.nn.LayerNorm
LLAMA_MODEL = transformers.models.llama.modeling_llama.LlamaForCausalLM
LLAMA_LAYER = transformers.models.llama.modeling_llama.LlamaDecoderLayer
LLAMA_NORM = transformers.models.llama.modeling_llama.LlamaRMSNorm
MISTRAL_MODEL = transformers.models.mistral.modeling_mistral.MistralForCausalLM
MISTRAL_LAYER = transformers.models.mistral.modeling_mistral.MistralDecoderLayer
MISTRAL_NORM = transformers.models.mistral.modeling_mistral.MistralRMSNorm
QWEN2_MODEL = transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM
QWEN2_LAYER = transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer
QWEN2_NORM = transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm
INTERNLM2_MODEL = None
INTERNLM2_LAYER = None
INTERNLM2_NORM = None
INTERNVL2_MODEL = None
INTERNVL2_LAYER = None
INTERNVL2_NORM = None
INTERNVL2_L_MODEL = None
INTERNVL2_L_LAYER = None
INTERNVL2_L_NORM = None
LLAVA_MODEL = None
LLAVA_LAYER = None
LLAVA_NORM = None
LLAVA_OV_MODEL = None
LLAVA_OV_LAYER = None
LLAVA_OV_NORM = None

def model_type_extractor(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    elif isinstance(model, OPT_MODEL):
        return OPT_MODEL
    elif isinstance(model, MISTRAL_MODEL):
        return MISTRAL_MODEL
    elif isinstance(model, QWEN2_MODEL):
        return QWEN2_MODEL
    elif model.config.architectures[0] == 'InternLM2ForCausalLM':
        global INTERNLM2_MODEL,INTERNLM2_LAYER,INTERNLM2_NORM
        INTERNLM2_MODEL = model.__class__
        INTERNLM2_LAYER = model.model.layers.__class__
        INTERNLM2_NORM = model.model.norm.__class__
        return INTERNLM2_MODEL
    elif model.__class__.__name__ in ["InternVLChatModel", "InternVL2"]:
        global INTERNVL2_MODEL,INTERNVL2_LAYER,INTERNVL2_NORM
        INTERNVL2_MODEL = model.__class__
        INTERNVL2_LAYER = model.language_model.model.layers.__class__
        INTERNVL2_NORM = model.language_model.model.norm.__class__
        return INTERNVL2_MODEL
    elif model.__class__.__name__ == "InternVL2ForCausalLM":
        global INTERNVL2_L_MODEL,INTERNVL2_L_LAYER,INTERNVL2_L_NORM
        INTERNVL2_L_MODEL = model.__class__
        INTERNVL2_L_LAYER = model.model.layers.__class__
        INTERNVL2_L_NORM = model.model.norm.__class__
        return INTERNVL2_L_MODEL
    elif model.__class__.__name__ == "LlavaQwenForCausalLM":
        global LLAVA_OV_MODEL,LLAVA_OV_LAYER,LLAVA_OV_NORM
        LLAVA_OV_MODEL = model.__class__
        LLAVA_OV_LAYER = model.model.layers.__class__
        LLAVA_OV_NORM = model.model.norm.__class__
        return LLAVA_OV_MODEL
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        global LLAVA_MODEL,LLAVA_LAYER,LLAVA_NORM
        LLAVA_MODEL = model.__class__
        LLAVA_LAYER = model.model.layers.__class__
        LLAVA_NORM = model.model.norm.__class__
        return LLAVA_MODEL

    else:
        raise ValueError(f'Unknown model type {model}')

def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization! 
    pass

def get_rope_function_name(model):
    if isinstance(model, (LLAMA_MODEL, MISTRAL_MODEL, QWEN2_MODEL, LLAVA_OV_MODEL, LLAVA_MODEL, INTERNVL2_MODEL)):
        return "apply_rotary_pos_emb"
    raise NotImplementedError


def get_layers(model):
    if isinstance(model, OPT_MODEL):
        return model.model.decoder.layers
    if isinstance(model, LLAMA_MODEL):
        return model.model.layers
    if isinstance(model, MISTRAL_MODEL):
        return model.model.layers
    if isinstance(model, QWEN2_MODEL):
        return model.model.layers
    if isinstance(model, INTERNLM2_MODEL):
        return model.model.layers
    if isinstance(model, INTERNVL2_MODEL) or isinstance(model, INTERNVL2_L_MODEL):
        return model.language_model.model.layers
    if isinstance(model, LLAVA_OV_MODEL) or isinstance(model, LLAVA_MODEL):
        return model.model.layers
    raise NotImplementedError


def get_llama(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.LlamaForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model



def get_opt(model_name):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.OPTForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                        low_cpu_mem_usage=True)
    model.seqlen = model.config.max_position_embeddings
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model


def get_model(
    model_name, hf_token=None
):
    if 'llama' in model_name:
        return get_llama(model_name, hf_token)
    elif 'opt' in model_name:
        return get_opt(model_name)
    else:
        raise ValueError(f'Unknown model {model_name}')


def get_model_type(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    elif isinstance(model, OPT_MODEL):
        return OPT_MODEL
    elif isinstance(model, MISTRAL_MODEL):
        return MISTRAL_MODEL
    elif isinstance(model, QWEN2_MODEL):
        return QWEN2_MODEL
    elif isinstance(model, INTERNVL2_MODEL):
        return INTERNVL2_MODEL
    elif isinstance(model, LLAVA_MODEL):
        return LLAVA_MODEL
    elif isinstance(model, INTERNVL2_L_MODEL):
        return INTERNVL2_L_MODEL
    elif isinstance(model, LLAVA_OV_MODEL):
        return LLAVA_OV_MODEL  
    else:
        raise ValueError(f'Unknown model type {model}')

def get_norm_type(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_NORM
    elif isinstance(model, OPT_MODEL):
        return OPT_NORM
    elif isinstance(model, MISTRAL_MODEL):
        return MISTRAL_NORM
    elif isinstance(model, QWEN2_MODEL):
        return QWEN2_NORM
    elif isinstance(model, INTERNVL2_MODEL):
        return INTERNVL2_NORM
    elif isinstance(model, LLAVA_OV_MODEL):
        return LLAVA_OV_NORM
    else:
        raise ValueError(f'Unknown model type {model}')
    
    
    
# def get_embeddings(model, model_type) -> list[torch.nn.Module]:
def get_embeddings(model, model_type):
    if model_type == LLAMA_MODEL or model_type == MISTRAL_MODEL or model_type == QWEN2_MODEL or model_type == LLAVA_MODEL or model_type == LLAVA_OV_MODEL :
        return [model.model.embed_tokens]
    elif model_type == INTERNLM2_MODEL:
        return [model.model.tok_embeddings]
    elif model_type == OPT_MODEL:
        return [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]
    elif model_type == INTERNVL2_MODEL or model_type == INTERNVL2_L_MODEL:
        if  hasattr(model.language_model, 'model') and hasattr(model.language_model.model,'tok_embeddings'):
            return [model.language_model.model.tok_embeddings]
        else:
            return [model.language_model.model.embed_tokens]
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_transformer_layers(model, model_type):
    if model_type == LLAMA_MODEL or model_type == MISTRAL_MODEL or model_type == QWEN2_MODEL or model_type == INTERNLM2_MODEL or model_type == LLAVA_MODEL or model_type == LLAVA_OV_MODEL:
        return [layer for layer in model.model.layers]
    elif model_type == OPT_MODEL:
        return [layer for layer in model.model.decoder.layers]
    elif model.__class__.__name__ == "InternVLChatModel" or model.__class__.__name__ == "InternVL2":
        return [layer for layer in model.language_model.model.layers]
    elif model.__class__.__name__ == "InternLM2ForCausalLM":
        return [layer for layer in model.model.layers]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    


def get_lm_head(model, model_type):
    if model_type == LLAMA_MODEL or model_type == MISTRAL_MODEL or model_type == QWEN2_MODEL:
        return model.lm_head
    elif model_type == OPT_MODEL:
        return model.lm_head
    elif model_type == INTERNLM2_MODEL:
        return model.output
    elif model_type == INTERNVL2_MODEL:
        return model.language_model.lm_head
    elif model_type == LLAVA_OV_MODEL:
        return model.lm_head
    else:
        raise ValueError(f'Unknown model type {model_type}')

def get_pre_head_layernorm(model, model_type):
    if model_type == LLAMA_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          LLAMA_NORM)
    elif model_type == QWEN2_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          QWEN2_NORM)
    elif model_type == MISTRAL_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          MISTRAL_NORM)
    elif model_type == OPT_MODEL:
        pre_head_layernorm = model.model.decoder.final_layer_norm
        assert pre_head_layernorm is not None
    elif model_type == INTERNLM2_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          INTERNLM2_NORM)
    elif model_type == INTERNVL2_MODEL:
        pre_head_layernorm = model.language_model.model.norm
        assert isinstance(pre_head_layernorm,
                          INTERNVL2_NORM)
    elif model_type == LLAVA_OV_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          LLAVA_OV_NORM)
    else:
        raise ValueError(f'Unknown model type {model_type}')
    return pre_head_layernorm

def get_mlp_bottleneck_size(model):
    model_type = get_model_type(model)
    if model_type == LLAMA_MODEL:
        return model.config.intermediate_size
    elif model_type == OPT_MODEL:
        return model.config.ffn_dim
    else:
        raise ValueError(f'Unknown model type {model_type}')

def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)


class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.variance_epsilon = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.ones(mean_dim))
        self.use_temporary_parameter = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_temporary_parameter:
            weight = self.temp_weight
        else:
            weight = self.weight

        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return x.to(input_dtype) * weight


def get_layer_io_save_path(args):
    return os.path.join(args.save_path, 'layer_io', f'{args.layer_idx:03d}.pt')

def capture_layer_io(model_type, layer, layer_input):
    def hook_factory(module_name, captured_vals, is_input):
        def hook(module, input, output):
            if is_input:
                captured_vals[module_name].append(input[0].detach().cpu())
            else:
                captured_vals[module_name].append(output.detach().cpu())
        return hook

    handles = []

    if model_type == LLAMA_MODEL:
        captured_inputs = {
            'k_proj': [],  # q_proj, v_proj has the same input as k_proj
            'o_proj': [],
            'gate_proj': [],  # up_proj has the same input as gate_proj
            'down_proj': []
        }

        captured_outputs = {
            'v_proj': [],
        }

        for name in captured_inputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))

    elif model_type == OPT_MODEL:
        captured_inputs = {
            'k_proj': [],  # q_proj, v_proj has the same input as k_proj
            'out_proj': [],
            'fc1': [],
            'fc2': []
        }
        captured_outputs = {
            'v_proj': [],
        }
        for name in captured_inputs.keys():
            # In OPT, fc1 and fc2 are directly contained in OPTDecoderLayer
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            # In OPT, fc1 and fc2 are directly contained in OPTDecoderLayer
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))
    else:
        raise ValueError(f'Unknown model type {model_type}')

    # Process each sequence in the batch one by one to avoid OOM.
    for seq_idx in range(layer_input.shape[0]):
        # Extract the current sequence across all dimensions.
        seq = layer_input[seq_idx:seq_idx + 1].to(utils.DEV)
        # Perform a forward pass for the current sequence.
        layer(seq)

    # After processing all sequences, concatenate the accumulated inputs for each sub-layer across the batch.
    for module_name in captured_inputs:
        captured_inputs[module_name] = torch.cat(captured_inputs[module_name], dim=0)
    for module_name in captured_outputs:
        captured_outputs[module_name] = torch.cat(captured_outputs[module_name], dim=0)

    # Cleanup.
    for h in handles:
        h.remove()

    return {
        'input': captured_inputs,
        'output': captured_outputs
    }


def mv_kv_cache(key_values, model=None, dev=None):
    '''
    move prefixed_key_values to corresponding device through full model or target_dec
    '''
    assert model is None or dev is None
    if key_values is None:
        return None
    key_values = list(key_values)
    if model is not None:
        layers = get_layers(model)
        for layer_index in range(len(key_values)):
            block_dev = next(layers[layer_index].parameters()).device
            key_values[layer_index] = list(key_values[layer_index])
            key_values[layer_index][0] = key_values[layer_index][0].to(block_dev)
            key_values[layer_index][1] = key_values[layer_index][1].to(block_dev)
            key_values[layer_index] = tuple(key_values[layer_index])
            
    if dev is not None:
        for layer_index in range(len(key_values)):
            key_values[layer_index] = list(key_values[layer_index])
            key_values[layer_index][0] = key_values[layer_index][0].to(dev)
            key_values[layer_index][1] = key_values[layer_index][1].to(dev)
            key_values[layer_index] = tuple(key_values[layer_index])
    key_values = tuple(key_values)
    return key_values


def get_kv_cache(prefixed_key_values, bs=1):
    if bs > 1:
        prefixed_key_values = kv_cache_repeat(prefixed_key_values, bs)
    if prefixed_key_values is not None:
        kv_cache = DynamicCache.from_legacy_cache(prefixed_key_values)
    else:
        kv_cache = None
    return kv_cache


def kv_cache_repeat(key_values, bs):
    '''
    bs 1 -> bs n
    '''
    if key_values is None:
        return None
    bs_key_values = {}
    for layer_index in range(len(key_values)):
        bs_key_values[layer_index] = list(key_values[layer_index])
        bs_key_values[layer_index][0] = bs_key_values[layer_index][0].repeat_interleave(bs, dim=0)
        bs_key_values[layer_index][1] = bs_key_values[layer_index][1].repeat_interleave(bs, dim=0)
        bs_key_values[layer_index] = tuple(bs_key_values[layer_index])
    return bs_key_values
    
#########################################################################
class CacheHook:
    def __init__(self, cachinglm):
        if cachinglm is None:
            self.dbdict = None
            return

        self.dbdict = cachinglm.dbdict

    def add_partial(self, attr, req, res):
        if self.dbdict is None:
            return
        hsh = hash_args(attr, req)
        self.dbdict[hsh] = res
        
class LM(abc.ABC):
    def __init__(self):
        self.cache_hook = CacheHook(None)

    @abstractmethod
    def loglikelihood(self, requests):
        """Compute log-likelihood of generating a continuation from a context.
        Downstream tasks should attempt to use loglikelihood instead of other
        LM calls whenever possible.

        :param requests: list
            A list of pairs (context, continuation)
            context: str
                Context string. Implementations of LM must be able to handle an
                empty context string.
            continuation: str
                The continuation over which log likelihood will be calculated. If
                there is a word boundary, the space should be in the continuation.
                For example, context="hello" continuation=" world" is correct.
        :return: list
            A list of pairs (logprob, isgreedy)
            logprob: float
                The log probability of `continuation`
            isgreedy:
                Whether `continuation` would be generated by greedy sampling from `context`
        """
        pass

    @abstractmethod
    def loglikelihood_rolling(self, requests):
        """Compute full log-likelihood of a string, with no truncation, for perplexity computation
        - We will use the full max context length of the model.
        - For inputs that exceed the max context length, we divide the tokenized string into chunks of up to
        the max context length.
        - IMPORTANT: Each document's loglikelihood/perplexity is computed *separately*, unlike other implementations
          which may simply concatenate multiple documents together.
        - IMPORTANT: We maximize the amount of context for each prediction. Specifically, for inputs that we break into
          multiple chunks, the last input will still a full-sized context.
          Example:
            Input tokens: [ 0 1 2 3 4 5 6 7 8 9 ]
            Prefix: EOT
            Max context length: 4
            Resulting input/prediction pairs:

                INPUT:  EOT   0   1   2
                PRED:     0   1   2   3

                INPUT:    3   4   5   6
                PRED:     4   5   6   7

                INPUT:    5   6   7   8
                PRED:             8   9

          Observe that:
            1. Each token is predicted exactly once
            2. For the last pair, we provide the full context, but only score the last two tokens

        :param requests: list
            A list of strings
            string: str
                String for which we are computing per-toke  loglikelihood
        :return: list
            A list of pairs (logprob, isgreedy)
            logprob: float
                The log probability of `continuation`
            isgreedy:
                Whether `continuation` would be generated by greedy sampling from `context`
        """
        pass

    # TODO: Add an optional max length
    @abstractmethod
    def greedy_until(self, requests):
        """Generate greedily until a stopping sequence

        :param requests: list
            A list of pairs (context, until)
            context: str
                Context string
            until: [str]
                The string sequences to generate until. These string sequences
                may each span across multiple tokens, or may be part of one token.
        :return: list
            A list of strings continuation
            continuation: str
                The generated continuation.
        """
        pass

    @classmethod
    def create_from_arg_string(cls, additional_config=None):
        additional_config = {} if additional_config is None else additional_config
        args = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args)

    def set_cache_hook(self, cache_hook):
        self.cache_hook = cache_hook


class BaseLM(LM):
    @property
    @abstractmethod
    def eot_token_id(self):
        pass

    @property
    @abstractmethod
    def max_length(self):
        pass

    @property
    @abstractmethod
    def max_gen_toks(self):
        pass

    @property
    @abstractmethod
    def batch_size(self):
        pass

    @property
    @abstractmethod
    def device(self):
        pass

    @abstractmethod
    def tok_encode(self, string: str):
        pass

    @abstractmethod
    def tok_decode(self, tokens: Iterable[int]):
        pass

    @abstractmethod
    def _model_generate(self, context, max_length, eos_token_id):
        pass

    @abstractmethod
    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        pass

    # subclass must implement properties vocab_size, eot_token_id, max_gen_toks, batch_size, device, max_length.
    # TODO: enforce this somehow

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            if context == "":
                # end of text as context
                context_enc = [self.eot_token_id]
            else:
                context_enc = self.tok_encode(context)

            continuation_enc = self.tok_encode(continuation)
            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: automatic batch size detection for vectorization

        loglikelihoods = []
        for (string,) in tqdm(requests):
            rolling_token_windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for
            # that
            string_nll = self._loglikelihood_tokens(
                rolling_token_windows, disable_tqdm=True
            )

            # discard is_greedy
            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []
        dataset_inps = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        # TODO: automatic (variable) batch size detection for vectorization
        re_ord = Reorderer(requests, _collate)
        for chunk in chunks(
            tqdm(re_ord.get_reordered(), disable=disable_tqdm), self.batch_size
        ):
            inps = []
            cont_toks_list = []
            inplens = []

            padding_length = None

            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                ).to(self.device)
                (inplen,) = inp.shape

                cont = continuation_enc

                # since in _collate we make sure length is descending, the longest is always the first one.
                padding_length = (
                    padding_length if padding_length is not None else inplen
                )

                # pad length from seq to padding_length
                inp = torch.cat(
                    [
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(
                            inp.device
                        ),  # [padding_length - seq]
                    ],
                    dim=0,
                )

                inps.append(inp.unsqueeze(0))  # [1, padding_length]
                cont_toks_list.append(cont)
                inplens.append(inplen)
            # import pdb; pdb.set_trace()
            batched_inps = torch.cat(inps, dim=0).to(
                self.device
            )  # [batch, padding_length

            # self.model = self.model.to(self.device)
            multi_logits = F.log_softmax(
                self._model_call(batched_inps), dim=-1
            ).cpu()  # [batch, padding_length, vocab]

            # dataset_inps.append(batched_inps)
            # dataset_logits = self._model_logits_on_dataset(dataset_inps)
            # iter = 0
            # for chunk in chunks(
            #         tqdm(re_ord.get_reordered(), disable=disable_tqdm), self.batch_size
            # ):
            #     multi_logits = dataset_logits[iter]
            #     iter+=1
            #     inps = []
            #     cont_toks_list = []
            #     inplens = []
            #
            #     padding_length = None
            #
            #     # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            #     # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            #     # again because vectorizing is annoying
            #
            #     # todo: check if we realy nead the following loop
            #     for _, context_enc, continuation_enc in chunk:
            #         # sanity check
            #         assert len(context_enc) > 0
            #         assert len(continuation_enc) > 0
            #         assert len(continuation_enc) <= self.max_length
            #
            #         # how this all works:
            #         #          CTX      CONT
            #         # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
            #         # gpt2    \               \
            #         # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
            #         # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice
            #
            #         # when too long to fit in context, truncate from the left
            #         inp = torch.tensor(
            #             (context_enc + continuation_enc)[-(self.max_length + 1):][:-1],
            #             dtype=torch.long,
            #         ).to(self.device)
            #         (inplen,) = inp.shape
            #
            #         cont = continuation_enc
            #
            #         # since in _collate we make sure length is descending, the longest is always the first one.
            #         padding_length = (
            #             padding_length if padding_length is not None else inplen
            #         )
            #
            #         # pad length from seq to padding_length
            #         inp = torch.cat(
            #             [
            #                 inp,  # [seq]
            #                 torch.zeros(padding_length - inplen, dtype=torch.long).to(
            #                     inp.device
            #                 ),  # [padding_length - seq]
            #             ],
            #             dim=0,
            #         )
            #
            #         inps.append(inp.unsqueeze(0))  # [1, padding_length]
            #         cont_toks_list.append(cont)
            #         inplens.append(inplen)

            for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(
                chunk, multi_logits, inps, inplens, cont_toks_list
            ):

                # Slice to original seq length
                contlen = len(cont_toks)
                logits = logits[inplen - contlen : inplen].unsqueeze(
                    0
                )  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(
                    0
                )  # [1, seq]
                # import pdb; pdb.set_trace()
                max_equal = (greedy_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                    -1
                )  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = (float(logits.sum()), bool(max_equal))

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                res.append(answer)

        return re_ord.get_original(res)

    def greedy_until(self, requests):
        print("greedy utils in base...")
        # TODO: implement fully general `until` that handles until that are
        #       multiple tokens or that span multiple tokens correctly

        # TODO: extract to TokenizedLM?
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = Reorderer(requests, _collate)

        for context, until in tqdm(re_ord.get_reordered()):
            if isinstance(until, str):
                until = [until]

            (primary_until,) = self.tok_encode(until[0])

            context_enc = torch.tensor(
                [self.tok_encode(context)[self.max_gen_toks - self.max_length :]]
            ).to(self.device)

            cont = self._model_generate(
                context_enc, context_enc.shape[1] + self.max_gen_toks, primary_until
            )

            s = self.tok_decode(cont[0].tolist()[context_enc.shape[1] :])

            for term in until:
                s = s.split(term)[0]

            # partial caching
            self.cache_hook.add_partial("greedy_until", (context, until), s)

            res.append(s)

        return re_ord.get_original(res)


def make_disjoint_window(pair):
    """Takes output from get_rolling_token_windows and makes the context not overlap with the continuation"""
    a, b = pair
    return a[: len(a) - (len(b) - 1)], b


def hash_args(attr, args):
    dat = json.dumps([attr] + list(args))
    return hashlib.sha256(dat.encode("utf-8")).hexdigest()


def simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = args_string.split(",")
    args_dict = {}
    for arg in arg_list:
        k, v = arg.split("=")
        args_dict[k] = v
    return args_dict


def get_rolling_token_windows(token_list, prefix_token, max_seq_len, context_len):
    """
    - context_len allows for a rolling window context, allowing each prediction window to potentially
      condition on some context

    :param token_list: list
        List of tokens to be PREDICTED
    :param max_seq_len: int
        max_seq_len of model (or max_seq_len we want to use)
    :param context_len: int
        Amount of desired token context for prediction. Needs to be at least 1.
    :param prefix_token: token
        Dummy token like <eos> so the first token has something to condition on
    :return: generator
        Generator of tuples
            (input_tokens, pred_tokens)
        Note: Score only the last len(pred_tokens) logits of the LM
    """
    assert 1 <= context_len <= max_seq_len
    if not token_list:
        return
    # +1 offset, going from input->preds
    pred_len = max_seq_len - context_len + 1
    predicted = 0

    # Special handling for first window: predict all tokens
    first_seq_len = min(max_seq_len, len(token_list))
    yield ([prefix_token] + token_list[: first_seq_len - 1], token_list[:first_seq_len])
    predicted += first_seq_len

    while predicted < len(token_list):
        window_pred_len = min(len(token_list) - predicted, pred_len)
        window_end = predicted + window_pred_len

        yield (
            token_list[window_end - max_seq_len - 1 : window_end - 1],
            token_list[window_end - window_pred_len : window_end],
        )
        predicted += window_pred_len


class Reorderer:
    def __init__(self, arr, fn):
        self.size = len(arr)
        arr = list(enumerate(arr))
        arr = group(arr, lambda x: fn(x[1]))
        arr = [([y[0] for y in x], x[0][1]) for x in arr]
        arr.sort(key=lambda x: fn(x[1]))

        self.arr = arr

    def get_reordered(self):
        return [x[1] for x in self.arr]

    def get_original(self, newarr):
        res = [None] * self.size
        cov = [False] * self.size

        for (inds, _), v in zip(self.arr, newarr):
            for ind in inds:
                res[ind] = v
                cov[ind] = True

        assert all(cov)

        return res


def join_iters(iters):
    for iter in iters:
        yield from iter


def chunks(iter, n):
    arr = []
    for x in iter:
        arr.append(x)
        if len(arr) == n:
            yield arr
            arr = []

    if arr:
        yield arr


def group(arr, fn):
    res = collections.defaultdict(list)

    for ob in arr:
        res[fn(ob)].append(ob)

    return list(res.values())


class LMClass(BaseLM):
    def __init__(self, args, model, tokenizer):

        super().__init__()
        self.model_name = args.model_name
        self.batch_size_per_gpu = args.eval_batch_size
        # self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # config = AutoConfig.from_pretrained(args.model_path,trust_remote_code=True)
        # self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False,legacy=False,trust_remote_code=True)
        # dtype = torch.float16 if not args.use_fp32 else torch.float32
        # self.model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, device_map='cpu',torch_dtype=dtype,trust_remote_code=True)
        # self.seqlen = self.model.config.max_position_embeddings
        self.tokenizer = tokenizer
        self.model = model
        self._device = self.model.lm_head.weight.device
        self.seqlen = self.model.config.max_position_embeddings
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        print("vocab size: ", self.vocab_size)

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        print("max_gen_toks fn")
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_encode_batch(self, strings):
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():

            return self.model(inps)["logits"]

    def model_batched_set(self, inps):
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(
                self._model_call(batch), dim=-1
            ).cpu()  # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
