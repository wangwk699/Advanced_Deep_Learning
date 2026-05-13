# Advanced Deep Learning 大作业：基于 CLIP 的开放集 Zero-shot 食物图像识别

本项目面向硕士生大作业要求，实现一个基于 Foundation Model 的食物图像与文本开放集 zero-shot 识别系统。系统使用 CLIP 的图文对齐能力，不对目标数据进行有监督训练或微调。

## 1. 任务说明

目标是完成 5 类食物图像的开放集 zero-shot 识别：

- 当输入图像属于已知 5 类之一时，输出对应的已知类别；
- 当输入图像不属于已知 5 类时，先检测为 `unknown`，再在预设的 Food-101 unknown 候选标签中输出具体预测类别。

## 2. 类别设置

### 已知类 known classes

本地 `Food_Multimodal` 数据集中的 5 类：

| 英文标签 | 中文类别 |
| --- | --- |
| `baozi` | 包子 |
| `cupcake` | 小蛋糕 |
| `french_fries` | 薯条 |
| `rice` | 米饭 |
| `youtiao` | 油条 |

### 未知类 unknown classes

从 Food-101 中选取与已知类不重合的 5 类作为 unknown 候选类别：

| 英文标签 | 中文类别 |
| --- | --- |
| `hamburger` | 汉堡包 |
| `pizza` | 披萨 |
| `sushi` | 寿司 |
| `steak` | 牛排 |
| `ice_cream` | 冰淇淋 |

## 3. 方法简介

本项目使用 `openai/clip-vit-base-patch32` 作为基础模型。实现流程如下：

1. 为每个类别构造文本 prompt，例如 `a photo of french fries, a type of food`。
2. 使用 CLIP 文本编码器提取类别文本特征。
3. 使用 CLIP 图像编码器提取输入图像特征。
4. 计算图像特征与类别文本特征的余弦相似度。
5. 对已知类进行 zero-shot closed-set 分类。
6. 在开放集评估中，若图像对所有已知类的最大置信度低于阈值，则判定为 `unknown`。
7. 对判定为 unknown 的图像，再与 Food-101 unknown 候选标签文本特征计算相似度，输出具体 unknown 类别预测。

整个过程没有训练分类头，也没有对 CLIP 或目标数据进行有监督微调。

## 4. 项目文件说明

```text
Advanced_Deep_Learning_modified/
├── main.py                  # 主程序：模型加载、zero-shot 评估、开放集评估、结果保存
├── data_loader.py           # 本地 Food_Multimodal 与 Food-101 数据加载
├── arguments.py             # 命令行参数定义
├── requirements.txt         # 稳定依赖版本
├── run.sh                   # 单次实验运行脚本
├── run_threshold_sweep.sh   # 阈值扫描脚本
├── README.md                # 实验说明文档
└── .gitignore
```

## 5. 环境配置

建议使用独立 conda 环境，例如：

```bash
conda create -n deep python=3.10 -y
conda activate deep
pip install -r requirements.txt
```

如果服务器需要安装 CUDA 11.8 对应的 PyTorch，也可以使用：

```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.38.2 datasets==2.18.0 numpy==1.26.4 Pillow pandas tqdm scikit-learn accelerate
```

说明：`requirements.txt` 中锁定了 `numpy==1.26.4` 和 `transformers==4.38.2`，用于避免 NumPy 2.x 与旧版 PyTorch、过新版 transformers 与 `torch==2.2.0` 之间的兼容性冲突。

## 6. 数据组织方式

本地已知类数据集应放在项目目录下的 `Food_Multimodal` 中，结构示例：

```text
Food_Multimodal/
├── train/
│   ├── metadata.csv
│   └── images/
├── val/
│   ├── metadata.csv
│   └── images/
└── test/
    ├── metadata.csv
    └── images/
```

`metadata.csv` 至少需要包含：

| 字段 | 含义 |
| --- | --- |
| `image_path` | 图像文件名或相对路径 |
| `label` | 类别标签，例如 `baozi`、`rice` |
| `text` | 可选文本描述 |

Food-101 unknown 数据通过 HuggingFace `datasets` 自动下载，运行服务器需要能访问 HuggingFace 数据集缓存或外网。

## 7. 运行方式

### 单次运行

```bash
bash run.sh
```

或手动运行：

```bash
python main.py \
  --data_root ./Food_Multimodal \
  --model_name openai/clip-vit-base-patch32 \
  --image_size 224 \
  --threshold 0.3 \
  --use_energy_score false \
  --temperature 0.07 \
  --max_unknown_per_class 50 \
  --output_dir ./results \
  --per_device_eval_batch_size 32 \
  --dataloader_num_workers 4 \
  --report_to none
```

### 指定 GPU

如果服务器有多张 GPU，可以在运行前设置：

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh
```

或在 `run.sh` 中取消注释：

```bash
export CUDA_VISIBLE_DEVICES=0
```

此时程序中的 `torch.device("cuda")` 会使用当前进程可见 GPU 中的 `cuda:0`。

### 阈值扫描

开放集阈值会显著影响 known/unknown 的取舍，可以运行：

```bash
bash run_threshold_sweep.sh
```

结果会分别保存到：

```text
results/threshold_0.2/
results/threshold_0.25/
...
```

## 8. 输出结果

程序会在 `output_dir` 中生成：

```text
results.txt
```

其中包括：

- 实验参数配置；
- 已知类 closed-set zero-shot 验证集准确率；
- 已知类 closed-set zero-shot 测试集准确率；
- 已知类分类报告；
- 开放集 known/unknown 检测准确率；
- 已知类召回率；
- 未知类检测率；
- unknown 候选类别预测准确率；
- unknown 具体预测类别分布；
- 开放集分类报告与混淆矩阵。

## 9. 主要修改说明

相较于初始版本，本版本做了以下完善：

1. 修复 `classification_report` 类别数量与 `target_names` 不一致的潜在错误。
2. 将 closed-set 已知类评估和 open-set known/unknown 评估分开输出。
3. 增加 unknown 样本的具体类别预测与 `unknown_class_accuracy`。
4. 保存 unknown 类别预测分布，便于检查系统是否真正输出 unknown 备选标签。
5. 锁定依赖版本，避免 NumPy 2.x、PyTorch、transformers 版本冲突。
6. 补充 README 说明文档，便于压缩包提交和复现实验。

## 10. 注意事项

- 本项目没有对目标数据进行有监督训练或微调，符合 zero-shot 要求。
- 开放集阈值 `threshold` 需要结合实际验证结果调整。
- 如果 HuggingFace Food-101 下载失败，可以提前在联网环境缓存数据集，或将 Food-101 子集改成本地读取。
- 如果显存不足，可以减小 `--per_device_eval_batch_size`。
