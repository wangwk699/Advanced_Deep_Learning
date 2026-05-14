#!/bin/bash
# 阈值扫描脚本 - 寻找最佳开放集识别阈值

export CUDA_VISIBLE_DEVICES=1
DATA_ROOT="./Food_Multimodal"

for threshold in 0.2 0.25 0.3 0.35 0.4 0.45 0.5
do
  echo "==========================================="
  echo "正在评估 threshold=${threshold}"
  echo "==========================================="

  python main.py \
    --data_root ${DATA_ROOT} \
    --model_name openai/clip-vit-base-patch32 \
    --image_size 224 \
    --threshold ${threshold} \
    --use_energy_score false \
    --temperature 0.07 \
    --max_unknown_per_class 50 \
    --output_dir "./results/threshold_${threshold}" \
    --per_device_eval_batch_size 32 \
    --dataloader_num_workers 4 \
    --report_to none \
    --run_name clip_food_threshold_${threshold} \
    --seed 42
done

echo "所有阈值扫描完成!"
