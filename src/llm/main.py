import os
import sys
import random
import numpy as np
import torch
import time
import pickle
from utils.data_utils import get_loaders, test_ppl
from quantize.block_ap import block_ap
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from quantize.int_linear_real import load_quantized_model
from accelerate import infer_auto_device_map, dispatch_model
from utils.quant_utils import wrap_to_quant_model, init_weight_quantizer, init_input_quantizer, register_online_had,  init_k_quantizer, init_v_quantizer,get_quant_config, model_bit_refactor,set_quant_state
from utils import train_utils
from awq.quantize.pre_quant import run_awq, apply_clip_multi_bit, apply_scale
import utils.model_utils as model_utils
import utils.rotation_utils as rotation_utils
torch.backends.cudnn.benchmark = True

@torch.no_grad()
def evaluate(model, tokenizer, args, logger):
    if "opt" in args.model_name.lower():
        block_class_name = model.model.decoder.layers[0].__class__.__name__ # opt
    elif "llama" in args.model_name.lower():
        block_class_name = model.model.layers[0].__class__.__name__ # llama
    
    
    device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
    model = dispatch_model(model, device_map=device_map)
    
    results_str=""
    if args.eval_ppl:
        datasets = ["wikitext2", "c4"]
        # datasets = ["wikitext2"]
        ppl_results = test_ppl(args, model, tokenizer, datasets)
        for dataset in ppl_results:
            logger.info(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')
            results_str += f"{ppl_results[dataset]:.2f} "
    
    if args.eval_tasks != "":
        results = {}
        from lm_eval import evaluator
        from pprint import pprint
        lm_model = model_utils.LMClass(args, model, tokenizer)
        t_results = evaluator.simple_evaluate(
            lm_model,
            tasks=args.eval_tasks,
            num_fewshot=0,
            limit=None,
        )
        results.update(t_results)
        logger.info(results)
        pprint(results)
    
def convert_dtype(model, dtype='torch.float16'):
    blocks = model.model.layers
    for i in range(len(blocks)):
        blocks[i] = blocks[i].to(dtype)   
    
def main():
    import argparse

    parser = argparse.ArgumentParser()
    # -----------------model setting------------------------------------
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument("--model_name", type=str, default=None,help="model name, for the saving of corresponding data cache")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="./log/", type=str, help="direction of logging file")
    parser.add_argument("--save_quant_dir", default=None, type=str, help="direction for saving quantization model")
    parser.add_argument("--real_quant", default=False, action="store_true",
                        help="use real quantization instead of fake quantization, can reduce memory footprint")
    parser.add_argument("--resume_quant", type=str, default=None,  help="model path of resumed quantized model")
    # -----------------quantization setting------------------------------------
    parser.add_argument("--wbits", type=int, default=2, help="quantization bits")
    parser.add_argument("--w_group_size", type=list, default=[128,128,128,-1,-1,-1,-1], help="quantization group size")
    parser.add_argument("--w_bit_list", type=list, default=[2,3,4,5,6,7,8], help="quantization bit list for weight")
    parser.add_argument("--rank_multiplier", type=int, default=1)
    parser.add_argument("--quant_type", type=str, default='weight_only', choices=['weight_only', 'weight_act'])
    parser.add_argument("--lora_type", type=str, default='partially_shared', choices=['partially_shared', 'fully_shared', 'independent'])
    parser.add_argument("--input_bits", type=int, default=4, help="quantization bits")
    parser.add_argument("--input_bit_list", type=list, default=[4,5,6,7,8], help="quantization bit list for input activation")
    parser.add_argument("--k_bits", type=int, default=4,help="")
    parser.add_argument("--v_bits", type=int, default=4,help="")
    parser.add_argument("--kv_group_size", type=int, default=128,help="default as head-wise")
    parser.add_argument('--lora-quant-mode', type=str, default='vanilla', 
                    choices=['vanilla', 'lora_on_scaled'],
                    help='How to integrate LoRA with quantization.')
    # ----------------- rotation setting------------------------------------
    parser.add_argument("--awq", action="store_true")
    parser.add_argument("--mae_loss", action="store_true")
    parser.add_argument("--huber_loss", action="store_true")
    parser.add_argument("--selective_merge", action="store_true")
    parser.add_argument("--uniform_fusion", action="store_true")
    parser.add_argument("--dump_awq", type=str, default=None, help="save the awq search results")
    parser.add_argument("--load_awq", type=str, default=None, help="load the awq search results")
    parser.add_argument("--pre_rotate", action="store_true")
    parser.add_argument("--rotate_mode", type=str, default='hadamard')
    parser.add_argument("--down_online_had", action="store_true")
    parser.add_argument("--qk_online_had", action="store_true")
    # -----------------calib dataset setting------------------------------------
    parser.add_argument("--calib_dataset",type=str,default="redpajama",
        choices=["wikitext2", "ptb", "c4", "mix", "redpajama", "pile"],
        help="Where to extract calibration data from.")
    parser.add_argument("--train_size", type=int, default=1024, help="Number of calibration data samples.")
    parser.add_argument("--val_size", type=int, default=64, help="Number of validation data samples.")
    parser.add_argument("--training_seqlen", type=int, default=2048, help="lenth of the training sequence.")
    # -----------------training setting------------------------------------
    parser.add_argument("--progressive_training", action="store_true")
    parser.add_argument("--clip_lr", type=float, default=1e-2, help="lr of quantization parameters (s and z)")
    parser.add_argument("--weight_lr", type=float, default=5e-6, help="lr of fp weights")
    parser.add_argument("--gd_lr", type=float, default=1e-4, help="gdloss learning rate")
    parser.add_argument("--min_lr_factor", type=float, default=100, help="min_lr = lr/min_lr_factor")
    parser.add_argument('--topk_token', type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2, help="batch size.")
    parser.add_argument("--loss_type", type=str, default="mse",help="")
    parser.add_argument("--token_dist_loss", action="store_true")
    parser.add_argument("--wd", type=float, default=0,help="weight decay")
    parser.add_argument("--use_fp32", action="store_true")
    parser.add_argument("--reduce_memory", default=False, action="store_true") 
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--early_stop", type=int, default=0,help="early stoping after validation loss do not decrease")
    parser.add_argument("--constant_wlr", action="store_true")
    # -----------------evaluation setting------------------------------------
    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="lenth of the training sequence.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true",help="evaluate perplexity on wikitext2 and c4 with 2048 context length")
    parser.add_argument("--eval_tasks", type=str,default="", help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_memory", type=str, default="40GiB",help="The maximum memory of each GPU")


    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
        
    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_quant_dir:
        Path(args.save_quant_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = train_utils.create_logger(output_dir)
    logger.info(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # quant_config = get_quant_config(args)
    # train_utils.save_dict_as_json(quant_config, os.path.join(args.save_quant_dir, 'quant_config.json'))
    # return 
    if args.model_name is None:
        args.model_name = args.model_path.split('/')[-1]
        logger.info(f"model_name is None, setting as {args.model_name}")
        
    if args.resume_quant:
        # directly load quantized model for evaluation
        group_size = 128 if args.wbits in [2,3,4] else -1
        model, tokenizer = load_quantized_model(args.resume_quant,args.wbits, group_size)
        model_dtype = next(model.parameters()).dtype
    else:
        # load fp quantized model
        config = AutoConfig.from_pretrained(args.model_path,trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False,legacy=False,trust_remote_code=True)
        dtype = 'auto' # torch.float16 # if not args.use_fp32 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, device_map=device,torch_dtype=dtype,trust_remote_code=True)
        model_dtype = next(model.parameters()).dtype
        if args.awq and args.quant_type == 'weight_only':
            q_config = {
                "zero_point": True,  # by default True
                "q_group_size": args.w_group_size[0],  # whether to use group quantization
                }
            use_cache = model.config.use_cache
            model.config.use_cache = False
            if args.load_awq:
                print("Loading pre-computed AWQ results from", args.load_awq)
                awq_file = os.path.join(args.load_awq, 'awq.pt')
                awq_results = torch.load(awq_file, map_location=device)
                apply_scale(model, awq_results["scale"])
            else:   
                awq_results = run_awq(
                    model, tokenizer,
                    w_bit=args.wbits, q_config=q_config, args=args,
                    n_samples=128, seqlen=512,
                )
                if args.dump_awq:
                    Path(args.dump_awq).mkdir(parents=True, exist_ok=True)
                    save_file = os.path.join(args.dump_awq, 'awq.pt')
                    torch.save(awq_results, save_file)
                    print("AWQ results saved at", args.dump_awq)    
            model.config.use_cache = use_cache
            # import pdb; pdb.set_trace()
        if args.pre_rotate:
            rotation_utils.fuse_layer_norms(model)
            rotation_utils.rotate_model(model, rotate_mode=args.rotate_mode, online=args.down_online_had)
            convert_dtype(model, model_dtype)
            model.to(model_dtype)


        wrap_to_quant_model(model, args)
        # register on-line hadadamrd transformation
        if args.pre_rotate and args.down_online_had:
            register_online_had(model)
            # wrap rope for online_had and rope output capture
            rope_function_name = model_utils.get_rope_function_name(model)
            layers = model_utils.get_layers(model)
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                            layer.self_attn, 
                            rope_function_name, 
                            config=model.config,
                            online_had=args.qk_online_had)   
    
        # init weight quantizer
        if args.wbits < 16:
            logger.info('init weight quantizer')
            init_weight_quantizer(args, model)
            if args.awq:
                print("Loading Clip refactors")
                apply_clip_multi_bit(model, awq_results["clip"])
        # init input quantizer
        # if args.input_bits < 16:
        if args.quant_type == 'weight_act':
            logger.info('init input quantizer')
            init_input_quantizer(args, model)
        # init K quantizer
        if args.v_bits < 16:
            logger.info('init v quantizer')
            init_v_quantizer(args, model)
        # init V quantizer
        if args.k_bits < 16:
            # consistently init for wrap rope 
            logger.info('init k quantizer')
            init_k_quantizer(args, model)
        train_utils.cleanup_memory()
        
        for param in model.parameters():
            param.requires_grad = False
        

        if args.epochs > 0:
            assert args.wbits < 16 or args.input_bits < 16 or args.output_bits < 16
            logger.info("=== start quantization Training ===")
            tick = time.time()     
            # load calibration dataset
            cache_trainloader = f'{args.cache_dir}/dataloader_{args.model_name}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_train.cache'
            cache_valloader = f'{args.cache_dir}/dataloader_{args.model_name}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_val.cache'
            if os.path.exists(cache_trainloader) and os.path.exists(cache_valloader):
                trainloader = torch.load(cache_trainloader)
                logger.info(f"load trainloader from {cache_trainloader}")
                valloader = torch.load(cache_valloader)
                logger.info(f"load valloader from {cache_valloader}")
            else:
                trainloader, valloader = get_loaders(
                    args.calib_dataset,
                    tokenizer,
                    args.train_size,
                    args.val_size,
                    seed=args.seed,
                    seqlen=args.training_seqlen,
                )
                torch.save(trainloader, cache_trainloader)    
                torch.save(valloader, cache_valloader)    
            block_ap(model,args,trainloader,valloader,logger)
            logger.info(time.time() - tick)
    if not args.reduce_memory:
        logger.info(f"Convert model to {model_dtype}")
        convert_dtype(model, model_dtype)
    torch.cuda.empty_cache()
    train_utils.cleanup_memory()
    
    ## save model 
    if args.save_quant_dir:
        logger.info("start saving model")
        model.save_pretrained(args.save_quant_dir)  
        tokenizer.save_pretrained(args.save_quant_dir) 
        quant_config = get_quant_config(args)
        train_utils.save_dict_as_json(quant_config, os.path.join(args.save_quant_dir, 'quant_config.json'))
        logger.info(f"save model to {args.save_quant_dir} success")
        
    ## validation 
    if args.quant_type == 'weight_act':
        set_quant_state(model, True, True)
    elif args.quant_type == 'weight_only':
        set_quant_state(model, True, False)

    for b in [4,5,6,8]:
        model_bit_refactor(model, b)
        logger.info(f"INFO === Validation on {b}-Bit ===")
        evaluate(model, tokenizer,args,logger)
        torch.cuda.empty_cache()
        train_utils.cleanup_memory()


    

if __name__ == "__main__":
    print(sys.argv)
    main()
