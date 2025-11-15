
CUDA_VISIBLE_DEVICES=0 python main.py \
--model_path /model/ModelZoo/LLM/llama2/Llama-2-7b-hf/ \
--model_name Llama-2-7b \
--calib_dataset c4 \
--train_size 128 \
--training_seqlen 2048 \
--k_bits 16 \
--v_bits 16 \
--pre_rotate \
--down_online_had \
--qk_online_had \
--output_dir ./log/Multi-bit/WA/Llama-2-7b/ \
--rank_multiplier 16 \
--quant_type 'weight_act' \
--wbits 4 \
--input_bits 4 \
--clip_lr 1e-1 \
--weight_lr 7e-5 \
--epochs 15 \
--topk_token 0.5 \
--mae_loss \
--lora-quant-mode vanilla \
--selective_merge \
--eval_ppl \
--eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande

# CUDA_VISIBLE_DEVICES=0 python main.py \
# --model_path /model/ModelZoo/LLM/llama3/Meta-Llama-3-8B/ \
# --model_name Llama-3-8B \
# --calib_dataset c4 \
# --train_size 128 \
# --training_seqlen 2048 \
# --k_bits 16 \
# --v_bits 16 \
# --pre_rotate \
# --down_online_had \
# --qk_online_had \
# --output_dir ./log/Multi-bit/WA/Llama-3-8B/ \
# --rank_multiplier 16 \
# --quant_type 'weight_act' \
# --wbits 4 \
# --input_bits 4 \
# --clip_lr 1e-1 \
# --weight_lr 7e-5 \
# --epochs 15 \
# --topk_token 0.5 \
# --eval_ppl \
# --mae_loss \
# --lora-quant-mode vanilla \
# --selective_merge \
# --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande
# # --save_quant_dir ./cache/weight_act/llama3-8b \
