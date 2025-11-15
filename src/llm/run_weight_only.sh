
CUDA_VISIBLE_DEVICES=0 python main.py \
--model_path /model/ModelZoo/LLM/llama2/Llama-2-7b-hf/  \
--model_name Llama-2-7b \
--calib_dataset c4 \
--train_size 128 \
--training_seqlen 2048 \
--k_bits 16 \
--v_bits 16 \
--kv_group_size 128 \
--output_dir ./log/Multi-bit/WO/Llama-2-7b/ \
--rank_multiplier 16 \
--quant_type 'weight_only' \
--wbits 2 \
--clip_lr 2e-2 \
--weight_lr 7e-4 \
--topk_token 0.5 \
--eval_ppl \
--awq \
--dump_awq ./cache/llama2-7b/ \
--selective_merge \
--epochs 15 \
--mae_loss \
--lora-quant-mode vanilla
# --save_quant_dir ./cache/weight_only/llama2-7b/