#!/bin/bash
# 运行脚本 - 基于 CLIP 的开放集 Zero-shot 食物图像识别

# 指定使用的 GPU 设备
export CUDA_VISIBLE_DEVICES=0

# 数据集路径（根据实际环境修改）
DATA_ROOT="./Food_Multimodal"

# 运行主程序
python main.py \
    --data_root ${DATA_ROOT} \
    --model_name openai/clip-vit-base-patch32 \
    --image_size 224 \
    --threshold 0.3 \
    --use_energy_score false \
    --temperature 0.07 \
    --output_dir ./results \
    --per_device_eval_batch_size 32 \
    --dataloader_num_workers 4 \
    --report_to none \
    --run_name clip_food_openset \
    --seed 42
