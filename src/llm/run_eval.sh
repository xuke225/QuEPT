
CUDA_VISIBLE_DEVICES=0 python eval.py \
--quant_model_path ./cache/weight_act/llama3-8b \
--output_dir ./log/Multi-bit/Wa/Llama3-8b/ \
--model_name Llama-3-8B \
--eval_ppl \
--eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
