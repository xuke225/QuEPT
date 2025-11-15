date=20251109
path=quept-vit
for model in vit_small deit_small swin_small
    do
        CUDA_VISIBLE_DEVICES=0 python test_quant.py --model $model \
            --dataset "/data/cls_dataset/ImageNet/" \
            --output_dir ./log/$date/$path/$model/$mixer_level/ \
            --wbits 4 --abits 4 \
            --rec_batch_size 32 --seed 42 \
            --epochs 20 \
            --lwc \
            --scaleLinear \
            --mixer_level Selective-Merge \
            --topk_token 0.5 \
            --mae 
done