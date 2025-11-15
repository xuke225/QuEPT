import os
import sys
import random
import numpy as np
import torch
import time
from quantize.block_ap import block_ap
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from quantize.int_linear_real import load_quantized_model
from accelerate import infer_auto_device_map, dispatch_model
from utils.quant_utils import (wrap_to_quant_model, wrap_to_quant_vlm_model, init_weight_quantizer,
                                init_input_quantizer, register_online_had, init_k_quantizer, 
                                init_v_quantizer,get_quant_config, model_bit_refactor,set_quant_state,
                                weight_act_bit_refactor)
from utils import train_utils
from awq.quantize.pre_quant import run_awq_llm, run_awq_vlm, apply_clip_multi_bit, apply_scale, get_blocks
import utils.model_utils as model_utils
import utils.rotation_utils as rotation_utils
torch.backends.cudnn.benchmark = True

# vlm import
from lmms_eval.models import get_model
from models import get_process_model
from calibration.coco_vl import get_multimodal_calib_dataset
import argparse
from functools import partial
from eval_vlm import cli_evaluate

def _int_or_none_list_arg_type(min_len: int, max_len: int, defaults: str, value: str, split_char: str = ","):
    def parse_value(item):
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{item} is not an integer or None")

    items = [parse_value(v) for v in value.split(split_char)]
    num_items = len(items)

    if num_items == 1:
        # Makes downstream handling the same for single and multiple values
        items = items * max_len
    elif num_items < min_len or num_items > max_len:
        raise argparse.ArgumentTypeError(f"Argument requires {max_len} integers or None, separated by '{split_char}'")
    elif num_items != max_len:
        print(f"Argument requires {max_len} integers or None, separated by '{split_char}'. " "Missing values will be filled with defaults.")
        default_items = [parse_value(v) for v in defaults.split(split_char)]
        items.extend(default_items[num_items:])  # extend items list with missing defaults

    return items

def convert_dtype(model, dtype='torch.float16'):
    blocks = get_blocks(model.model)
    for i in range(len(blocks)):
        blocks[i] = blocks[i].to(dtype)
    
def main():
    import argparse

    parser = argparse.ArgumentParser()
    # -----------------model setting------------------------------------
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument("--model_name", type=str, default=None,help="model name, for the saving of corresponding data cache")
    parser.add_argument("--model_args", type=str, default="/model/llava-onevision-qwen2-7b-ov/",help="Control parameters passed to the model constructor")#
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
    parser.add_argument("--lora_type", type=str, default='bit_part_share', choices=['all_shared', 'bit_ind','bit_part_share','shared_up'])
    parser.add_argument("--quant_type", type=str, default='weight_only', choices=['weight_only', 'weight_act'])
    parser.add_argument("--input_bits", type=int, default=4, help="quantization bits")
    parser.add_argument("--input_bit_list", type=list, default=[4,5,6,7,8], help="quantization bit list for input activation")
    parser.add_argument("--k_bits", type=int, default=4,help="")
    parser.add_argument("--v_bits", type=int, default=4,help="")
    parser.add_argument("--kv_group_size", type=int, default=128,help="default as head-wise")
    # ----------------- rotation setting------------------------------------
    parser.add_argument("--awq", action="store_true")
    parser.add_argument("--smoothquant", action="store_true")
    parser.add_argument("--dump_awq", type=str, default=None, help="save the awq search results")
    parser.add_argument("--load_awq", type=str, default=None, help="load the awq search results")
    parser.add_argument("--pre_rotate", action="store_true")
    parser.add_argument("--rotate_mode", type=str, default='hadamard')
    parser.add_argument("--down_online_had", action="store_true")
    parser.add_argument("--qk_online_had", action="store_true")
    # -----------------calib dataset setting------------------------------------
    parser.add_argument("--calib_dataset",type=str,default="redpajama",
        choices=["wikitext2", "ptb", "c4", "mix", "redpajama", "pile", "coco", "pileval"],
        help="Where to extract calibration data from.")
    parser.add_argument("--train_size", type=int, default=1024, help="Number of calibration data samples.")
    parser.add_argument("--val_size", type=int, default=64, help="Number of validation data samples.")
    parser.add_argument("--training_seqlen", type=int, default=2048, help="lenth of the training sequence.")
    # -----------------training setting------------------------------------
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--progressive_training", action="store_true")
    parser.add_argument("--clip_lr", type=float, default=1e-2, help="lr of quantization parameters (s and z)")
    parser.add_argument("--weight_lr", type=float, default=5e-6, help="lr of fp weights")
    parser.add_argument("--min_lr_factor", type=float, default=100, help="min_lr = lr/min_lr_factor")
    parser.add_argument('--topk_token', type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--loss_type", type=str, default="mse",help="")
    parser.add_argument("--wd", type=float, default=0,help="weight decay")
    parser.add_argument("--use_fp32", action="store_true")
    parser.add_argument("--reduce_memory", default=False, action="store_true") 
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--early_stop", type=int, default=0,help="early stoping after validation loss do not decrease")
    parser.add_argument("--constant_wlr", action="store_true")
    # -----------------vlm setting------------------------------------
    parser.add_argument("--data_path", type=str, default="/data/vlm_dataset/ShareGPT4V/data/sharegpt4v/coco_caption.json")
    parser.add_argument("--text_data_path", type=str, default="",help="You should specify this arg if you want to use interleave_format")
    parser.add_argument("--image_folder", type=str, default="/data/vlm_dataset/ShareGPT4V/data/")
    parser.add_argument("--few_shot_format", type=bool, default=False,help="if True, concat two samples to simulate few shot")
    parser.add_argument("--interleave_format", type=bool, default=False,help="if True, insert pure text data between two image-text pairs to simulate interleave data")
    parser.add_argument("--loss_mode", default="mae", choices=["mae", "mse"])
    parser.add_argument("--quant_vision", action="store_true")
    # -----------------vlm eval setting------------------------------------
    parser.add_argument("--tasks", default=None, help="To get full list of tasks, use the command lmms-eval --tasks list")
    parser.add_argument("--num_fewshot", type=int, default=None, help="Number of examples in few-shot context")
    parser.add_argument("--eval_batch_size","-b",type=str, default=1, metavar="auto|auto:N|N", help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",)
    parser.add_argument("--max_batch_size", type=int, default=None, metavar="N", help="Maximal batch size to try with --batch_size auto.",)    
    parser.add_argument("--output_path", default=None, type=str, metavar="= [dir/file.jsonl] [DIR]", help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",)
    parser.add_argument("--limit", type=float, default=None, help="Limit the number of examples per task. " "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--use_cache", "-c", type=str, default=None, metavar="DIR", help="A path to a sqlite db file for caching model responses. `None` if not caching.",)
    parser.add_argument("--cache_requests", type=str, default=None, choices=["true", "refresh", "delete"], help="Speed up evaluation by caching the building of dataset requests. `None` if not caching.",)
    parser.add_argument("--check_integrity", action="store_true", help="Whether to run the relevant part of the test suite for the tasks",)
    parser.add_argument("--write_out", "-w", action="store_true", default=False, help="Prints the prompt for the first few documents.",)
    parser.add_argument("--log_samples", action="store_true", default=True, help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis",)
    parser.add_argument("--wandb_log_samples", action="store_true", default=False, help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis to Weights and Biases",)
    parser.add_argument("--log_samples_suffix", type=str, default="model_outputs", help="Specify a suffix for the log_samples file name.",)
    parser.add_argument("--system_instruction", type=str, default=None, help="System instruction to be used in the prompt",)
    parser.add_argument("--apply_chat_template", action="store_true", default=False, help="If True, applies the chat template to the prompt",)
    parser.add_argument("--fewshot_as_multiturn", action="store_true", default=False, help="If True, uses the fewshot as a multi-turn conversation",)
    parser.add_argument("--show_config", action="store_true", default=False, help="If True, shows the the full config of all tasks at the end of the evaluation.",)
    parser.add_argument("--include_path", type=str, default=None, help="Additional path to include if there are external tasks to include.",)
    parser.add_argument("--gen_kwargs", default="", help=("String arguments for model generation on greedy_until tasks," " e.g. `temperature=0,top_k=0,top_p=0`"),)
    parser.add_argument("--verbosity", type=str, default="INFO", help="Log error when tasks are not registered.",)
    parser.add_argument("--wandb_args", default="", help="Comma separated string arguments passed to wandb.init, e.g. `project=lmms-eval,job_type=eval",)
    parser.add_argument("--timezone", default="Asia/Singapore", help="Timezone for datetime string, e.g. Asia/Singapore, America/New_York, America/Los_Angeles. You can check the full list via `import pytz; print(pytz.common_timezones)`",)
    parser.add_argument("--hf_hub_log_args", type=str, default="", help="Comma separated string arguments passed to Hugging Face Hub's log function, e.g. `hub_results_org=EleutherAI,hub_repo_name=lm-eval-results`",)
    parser.add_argument("--predict_only", "-x", action="store_true", default=False, help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",)
    parser.add_argument("--trust_remote_code", action="store_true", help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",)
    parser.add_argument("--process_with_media", type=bool, default=False, help="lmms-eval dataset arg",)
    default_seed_string = "0,1234,1234,1234"
    parser.add_argument("--seed", type=partial(_int_or_none_list_arg_type, 3, 4, default_seed_string),default=default_seed_string,  # for backward compatibility
        help=(
            "Set seed for python's random, numpy, torch, and fewshot sampling.\n"
            "Accepts a comma-separated list of 4 values for python's random, numpy, torch, and fewshot sampling seeds, "
            "respectively, or a single integer to set the same seed for all four.\n"
            f"The values are either an integer or 'None' to not set the seed. Default is `{default_seed_string}` "
            "(for backward compatibility).\n"
            "E.g. `--seed 0,None,8,52` sets `random.seed(0)`, `torch.manual_seed(8)`, and fewshot sampling seed to 52. "
            "Here numpy's seed is not set since the second value is `None`.\n"
            "E.g, `--seed 42` sets all four seeds to 42."
        ),
    )

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    random.seed(args.seed[0])
    np.random.seed(args.seed[0])
    torch.manual_seed(args.seed[0])
    torch.cuda.manual_seed(args.seed[0])
        
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
    else:
        # load fp quantized model
        ModelClass = get_model(args.model_name)
        lm = ModelClass.create_from_arg_string(
        args.model_args,
        {
            "batch_size": args.batch_size,
            "device": args.device,
        },
    )
        # Preprocess the MLLM here, use "lm._model" to get the fp16 mllm.
        Process_ModelClass = get_process_model(args.model_name)
        model = Process_ModelClass(lm._model, 
                                   lm._tokenizer, 
                                   lm.processor if hasattr(lm, 'processor') else None)
        model_dtype = next(model.model.parameters()).dtype
        # print(model.model)
        print(f"model_type:{model_dtype}")
        if args.awq and args.quant_type == 'weight_only':
            q_config = {
                "zero_point": True,  # by default True
                "q_group_size": args.w_group_size[0],  # whether to use group quantization
                }

            if args.load_awq:
                print("Loading pre-computed AWQ results from", args.load_awq)
                awq_file = os.path.join(args.load_awq, 'awq.pt')
                awq_results = torch.load(awq_file, map_location="cpu")
                apply_scale(model.model, awq_results["scale"])
            else:  
                awq_results = run_awq_vlm(
                    args,
                    model,
                    lm,
                    w_bit=args.wbits,
                    q_config=q_config,
                    auto_scale=True,
                    mse_range=True,
                    calib_data=args.calib_dataset,
                    loss_mode=args.loss_mode
                )
                if args.dump_awq:
                    Path(args.dump_awq).mkdir(parents=True, exist_ok=True)
                    save_file = os.path.join(args.dump_awq, 'awq.pt')
                    torch.save(awq_results, save_file)
                    print("AWQ results saved at", args.dump_awq)    
            # import pdb; pdb.set_trace()
        if args.smoothquant and args.quant_type == 'weight_act':
            if args.load_smoothquant:
                print("Loading pre-computed SmoothQuant results from", args.load_smoothquant)
        if args.pre_rotate:
            rotation_utils.fuse_layer_norms(model.model)
            rotation_utils.rotate_model(model.model, rotate_mode=args.rotate_mode, online=args.down_online_had)
            convert_dtype(model, model_dtype)
            model.model.to(model_dtype)
        wrap_to_quant_vlm_model(model.model, args)
        # register on-line hadadamrd transformation
        if args.pre_rotate and args.down_online_had:
            register_online_had(model.model)
            # wrap rope for online_had and rope output capture
            rope_function_name = model_utils.get_rope_function_name(model.model)
            layers = model_utils.get_layers(model.model)
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                            layer.self_attn, 
                            rope_function_name, 
                            config=model.model.config,
                            online_had=args.qk_online_had)   
    
        # init weight quantizer
        if args.wbits < 16:
            logger.info('init weight quantizer')
            init_weight_quantizer(args, model.model)
            if args.awq:
                print("Loading Clip refactors")
                apply_clip_multi_bit(model.model, awq_results["clip"])
        # init input quantizer
        # if args.input_bits < 16:
        if args.quant_type == 'weight_act':
            logger.info('init input quantizer')
            init_input_quantizer(args, model.model)
        # init K quantizer
        if args.v_bits < 16:
            logger.info('init v quantizer')
            init_v_quantizer(args, model.model)
        # init V quantizer
        if args.k_bits < 16:
            # consistently init for wrap rope 
            logger.info('init k quantizer')
            init_k_quantizer(args,model.model)
        train_utils.cleanup_memory()
        
        for param in model.model.parameters():
            param.requires_grad = False
        

        if args.epochs > 0:
            assert args.wbits < 16 or args.input_bits < 16 or args.output_bits < 16
            logger.info("=== start quantization Training ===")
            tick = time.time()     
            # load calibration dataset
            model.to_cuda()
            prompt_inputs, prompt_kwargs = get_multimodal_calib_dataset(
            data_path=args.data_path,
            image_folder=args.image_folder,
            model=model, 
            n_samples=args.train_size, 
            few_shot_format=args.few_shot_format,
            interleave_format=args.interleave_format, 
            text_data_path=args.text_data_path,
            shuffle=True,
        )
            block_ap(model, args, prompt_inputs, prompt_kwargs, logger)
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
        
    # Eval 
    if args.quant_type == 'weight_act':
        set_quant_state(model.model, True, True)
        for tasks in ["mmmu_val","ocrbench","textvqa_val","vizwiz_vqa_val","seedbench"]: 
            args.tasks = tasks
            args.log_samples_suffix = tasks
            logger.info(f"INFO === Eval on W4A4 ===")
            weight_act_bit_refactor(model.model, 4, 4)
            model.to_cuda()
            cli_evaluate(lm, args)
    elif args.quant_type == 'weight_only':
        set_quant_state(model.model, True, False)
        for tasks in ["mmmu_val","ocrbench","textvqa_val","vizwiz_vqa_val","seedbench"]: 
            args.tasks = tasks
            args.log_samples_suffix = tasks
            logger.info(f"INFO === Eval on W2 ===")
            model_bit_refactor(model.model, 2)
            model.to_cuda()
            cli_evaluate(lm, args)

    torch.cuda.empty_cache()
    train_utils.cleanup_memory()

if __name__ == "__main__":
    print(sys.argv)
    main()
