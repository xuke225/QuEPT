import os
import sys
import random
import numpy as np
import torch
import utils
import pickle
from utils import train_utils
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map
from utils.quant_utils import wrap_to_quant_model, init_weight_quantizer, init_input_quantizer, register_online_had, init_k_quantizer, init_v_quantizer,get_named_linears,set_op_by_name, set_quant_state, model_bit_refactor
import utils.model_utils as model_utils
import utils.rotation_utils as rotation_utils
from main import evaluate
from utils.train_utils import load_json_as_namespace,create_logger
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
import quantize.int_linear_fake as int_linear_fake
import quantize.int_linear_real as int_linear_real


torch.backends.cudnn.benchmark = True



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_model_path", type=str, help="model path of quantized model")
    parser.add_argument("--output_dir", default="./log/test", type=str, help="direction of logging file")
    parser.add_argument("--model_name", type=str, default=None,help="model name, for the saving of corresponding data cache")
    parser.add_argument("--real_quant", default=False, action="store_true",
                        help="use real quantization instead of fake quantization, can reduce memory footprint")
    parser.add_argument("--save_quant_dir", default=None, type=str, help="direction for saving quantization model")
    # parser.add_argument("--wbits", type=int, default=2, help="quantization bits")
    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="lenth of the training sequence.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true",help="evaluate perplexity on wikitext2 and c4 with 2048 context length")
    parser.add_argument("--eval_tasks", type=str,default="", help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_memory", type=str, default="40GiB",help="The maximum memory of each GPU")
    parser.add_argument("--lora_type", type=str, default='partially_shared', choices=['partially_shared', 'fully_shared', 'independent'])

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = create_logger(output_dir)

    quant_config = load_json_as_namespace(os.path.join(args.quant_model_path, 'quant_config.json'))
    logger.info(args)
    logger.info(quant_config)
    # init quantized model
    config = AutoConfig.from_pretrained(args.quant_model_path,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.quant_model_path, use_fast=False,legacy=False,trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(args.quant_model_path, config=config, device_map='cpu',torch_dtype=torch.float16,trust_remote_code=True)
    wrap_to_quant_model(model, quant_config)
    # register on-line hadadamrd transformation
    if quant_config.down_online_had:
        register_online_had(model)
    # wrap rope for online_had and rope output capture
    rope_function_name = model_utils.get_rope_function_name(model)
    layers = model_utils.get_layers(model)
    for layer in layers:
        rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn, 
                    rope_function_name, 
                    config=model.config,
                    online_had=quant_config.qk_online_had)   

        # init weight quantizer
        if quant_config.wbits < 16:
            print('init weight quantizer')
            init_weight_quantizer(quant_config, model)
        # init input quantizer
        # if args.input_bits < 16:
        if quant_config.quant_type == 'weight_act':
            print('init input quantizer')
            init_input_quantizer(quant_config, model)
        # init K quantizer
        if quant_config.v_bits < 16:
            print('init v quantizer')
            init_v_quantizer(quant_config, model)
        # init V quantizer
        if quant_config.k_bits < 16:
            # consistently init for wrap rope 
            print('init k quantizer')
            init_k_quantizer(quant_config, model)

    # model.tie_weights()
    device_map = infer_auto_device_map(model)
    print("Loading pre-computed quantized weights...")
    load_checkpoint_in_model(model,checkpoint=args.quant_model_path,device_map=device_map,dtype=torch.float16)
    model.half()    # to make sure same evaluation results with main
    
        ## validation 
    if quant_config.quant_type == 'weight_act':
        set_quant_state(model, True, True)
    elif quant_config.quant_type == 'weight_only':
        set_quant_state(model, True, False)
    for b in [4,5,6,8]:
        model_bit_refactor(model, b)
        logger.info(f"INFO === Validation on {b}-Bit ===")
        evaluate(model, tokenizer, args, logger)


    
    if args.real_quant:
        assert args.quant_type == 'weight_act' and args.k_bits>=16 and args.v_bits>=16, "only supprot for weight-only quantization"
        model.to('cpu')
        named_linears = get_named_linears(model, int_linear_fake.QuantLinear)
        for name, module in named_linears.items():
            scales, zeros, group_size = module.prepare_for_real(args.wbits)
            dim0 = module.weight.shape[0]
            scales = scales.view(dim0,-1).transpose(0,1).contiguous()
            zeros = zeros.view(dim0,-1).transpose(0,1).contiguous()
            q_linear = int_linear_real.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
            q_linear.pack(module.cpu(), args.wbits, scales.float().cpu(), zeros.float().cpu())
            set_op_by_name(model, name, q_linear)       
            logger.info(f"pack quantized {name} finished")
            del module        
        torch.cuda.empty_cache()
        if args.save_quant_dir:
            logger.info("start saving real quant model")
            model.save_pretrained(args.save_quant_dir)  
            tokenizer.save_pretrained(args.save_quant_dir) 
            logger.info(f"save model to {args.save_quant_dir} success")
        
        
if __name__ == "__main__":
    print(sys.argv)
    main()
