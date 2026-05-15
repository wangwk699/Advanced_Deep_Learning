#!/bin/bash
# 运行脚本 - 基于 CLIP 的统一候选标签 Zero-shot 食物图像识别

# 如需指定 GPU，可修改下面一行；如果不想固定 GPU，可以注释掉。
export CUDA_VISIBLE_DEVICES=0

# 数据集路径（根据实际环境修改）
DATA_ROOT="./Food_Multimodal"
Ensemble=false
INDEX=3
SAVE_DIR="./results/log2/prompt_ensemble_${Ensemble}/templates_index_${INDEX}"

# -m debugpy --listen 6001 --wait-for-client
python main.py \
  --data_root ${DATA_ROOT} \
  --model_name openai/clip-vit-base-patch32 \
  --image_size 224 \
  --prompt_ensemble "$Ensemble" \
  --templates_index "$INDEX" \
  --max_unknown_per_class 50 \
  --output_dir "$SAVE_DIR" \
  --per_device_eval_batch_size 32 \
  --dataloader_num_workers 4 \
  --report_to none \
  --run_name clip_food_unified_zero_shot \
  --seed 42
