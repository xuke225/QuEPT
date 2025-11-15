import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import pickle
import time
from utils import train_utils
from utils.data_utils import get_loaders
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
from dp import find_optimal_bit_allocation
from quantize.int_linear_fake import QuantLinear
from quantize.quant_norm import QuantRMSNorm


torch.backends.cudnn.benchmark = True

def pickle_dump_model(model,filepath):
    with open (filepath, 'wb') as f: 
        pickle.dump(model, f)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_bit_by_search_list(model, search_list):
    print(f"Set model block bitwidth by search_list: {search_list}")
    layers = model.model.layers
    cur_index = 0 
    while cur_index < len(layers): 
        sub_layer = layers[cur_index]
        set_quant_state(sub_layer,weight_quant=True, act_quant=False)
        model_bit_refactor(sub_layer, search_list[cur_index])
        cur_index += 1

def set_bit_by_search_list_per_layer(sub_layer, cur_index, search_list):
    print(f"Set model {cur_index} layer bitwidth to {search_list[cur_index]} bit")
    set_quant_state(sub_layer,weight_quant=True, act_quant=False)
    model_bit_refactor(sub_layer, search_list[cur_index])

def block_kl_on_cali_data(model, calidata, b_id, args, logger, device):
    layers = model.model.layers
    model.to(device)
    losses = AverageMeter()
    start_time = time.time()
    iters = len(calidata)// args.batch_size
    with torch.no_grad():
        for i in range(iters):
            inputs = torch.cat([calidata[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            set_quant_state(model, False, False)
            inputs = inputs.to(device)
            fp_outputs = model(inputs).logits
            sub_layer = layers[b_id]
            set_quant_state(sub_layer,weight_quant=True, act_quant=False)
            quant_outputs = model(inputs).logits
            loss = F.kl_div(F.log_softmax(quant_outputs, dim=1), F.softmax(
                fp_outputs, dim=1), reduction='batchmean')
            losses.update(loss.item(), inputs.size(0))
        current_time = time.time()
        logger.info(
            "KL Loss {:.4f}\ton {}th block\tTime {:.2f}s".format(
                float(losses.avg), b_id, (current_time - start_time)
            )
        )
    return losses.avg

def layer_kl_on_cali_data(model, sub_layer, calidata, l_id, args, logger, device):
    model.to(device)
    losses = AverageMeter()
    start_time = time.time()
    iters = len(calidata)// args.batch_size
    with torch.no_grad():
        for i in range(iters):
            inputs = torch.cat([calidata[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            set_quant_state(model, False, False)
            inputs = inputs.to(device)
            fp_outputs = model(inputs).logits
            set_quant_state(sub_layer,weight_quant=True, act_quant=False)
            quant_outputs = model(inputs).logits
            loss = F.kl_div(F.log_softmax(quant_outputs, dim=1), F.softmax(
                fp_outputs, dim=1), reduction='batchmean')
            losses.update(loss.item(), inputs.size(0))
        current_time = time.time()
        logger.info(
            "KL Loss {:.4f}\ton {}th layer\tTime {:.2f}s".format(
                float(losses.avg), l_id, (current_time - start_time)
            )
        )
    return losses.avg

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
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix", "redpajama", "pile"],
        help="Where to extract calibration data from.")
    parser.add_argument("--train_size", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--val_size", type=int, default=64, help="Number of validation data samples.")
    parser.add_argument("--training_seqlen", type=int, default=2048, help="lenth of the training sequence.")
    parser.add_argument("--batch_size",type=int, default=4)

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
            # print('init weight quantizer')
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

    set_quant_state(model, weight_quant=True, act_quant=False)

    sen_table = list()
    for i in range(2, 9):
        print("sensitivity for {}bit ".format(i))
        model_bit_refactor(model, i)
        temp_bit_list = []

        #layer-wise
        index = 0
        for layer in model.modules():
            if isinstance(layer, (QuantLinear,QuantRMSNorm)):
                temp_bit_list.append(layer_kl_on_cali_data(model, layer, trainloader, index, args, logger, device))
                index = index + 1
                     
        sen_table.append(temp_bit_list)

    save_path = args.output_dir + 'Sensitivity_look_up_table_per_layer' + \
        str(args.model_name)+'.pickle'
    # pickle_dump_model(sen_table, save_path)
    with open(save_path, 'rb') as file:
        sen_table = pickle.load(file)
        print(sen_table)

    bit_list = [2.25, 2.5, 2.75, 3, 4]
    config_list = []
    sen_list = []
    for b in bit_list:
        bits_config, min_sensitivity = find_optimal_bit_allocation(b, save_path, 288)
        config_list.append(bits_config)
        sen_list.append(min_sensitivity)

    dp_path = "./dp/dp_table_llama2_7b_trained_per_layer.pickle"
    pickle_dump_model(config_list, dp_path)
    print("Optimal bit configuration for each block:", config_list)
    print("Minimum total sensitivity:", sen_list)

    with open(dp_path, 'rb') as file:
        llama_dp_list =  pickle.load(file)

    for i in range(len(llama_dp_list)):
        logger.info(f"INFO === Validation on {llama_dp_list[i]} ===")
        cur_index = 0
        for layer in model.modules():
            if isinstance(layer, (QuantLinear,QuantRMSNorm)):
                set_bit_by_search_list_per_layer(layer, cur_index, llama_dp_list[i])
                cur_index = cur_index + 1
        evaluate(model, tokenizer,args,logger)
        torch.cuda.empty_cache()
        train_utils.cleanup_memory()
    
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