# Advanced Deep Learning 大作业：基于 CLIP 的统一候选标签 Zero-shot 食物图像识别

本项目面向硕士生大作业要求，实现一个基于 Foundation Model 的食物图像与文本开放集 zero-shot 识别系统。系统使用 CLIP 的图文对齐能力，不对目标数据进行有监督训练或微调。

与阈值拒识方案不同，本版本采用“统一候选标签”的第一种方案：将 5 个 known classes 与 5 个从 Food-101 选取的 unknown classes 合并为 10 个候选类别。对于每张输入图像，直接计算图像特征与全部候选文本特征之间的相似度，并选择相似度最高的类别作为预测结果。

## 1. 任务说明

目标是完成 5 类食物图像及若干 unknown 类别的 zero-shot 开放集测试：

- 当输入图像属于已知 5 类之一时，输出对应已知类别；
- 当输入图像不属于已知 5 类时，从预先选定的 Food-101 unknown 候选标签中输出预测类别；
- 不对目标数据进行有监督训练或微调。

本项目没有使用 known/unknown 阈值判断，也没有使用阈值扫描脚本。

## 2. 类别设置

### 已知类 known classes

本地 `Food_Multimodal` 数据集中的 5 类：

| 英文标签 | 对应类别 |
| --- | --- |
| `baozi` | 包子 |
| `cupcake` | 小蛋糕 |
| `french_fries` | 薯条 |
| `rice` | 米饭 |
| `youtiao` | 油条 |

### 未知类 unknown classes

从 Food-101 中选取与已知类不重合的 5 类作为 unknown 候选类别：

| 英文标签 | 对应类别 |
| --- | --- |
| `hamburger` | 汉堡包 |
| `pizza` | 披萨 |
| `sushi` | 寿司 |
| `steak` | 牛排 |
| `ice_cream` | 冰淇淋 |

最终候选标签集合为：

```text
[baozi, cupcake, french_fries, rice, youtiao, hamburger, pizza, sushi, steak, ice_cream]
```

## 3. 方法简介

本项目使用 `openai/clip-vit-base-patch32` 作为基础模型。主要流程如下：

1. 为 10 个候选类别构造英文文本 prompt，例如 `a photo of french fries, a type of food`。
2. 使用 CLIP 文本编码器提取类别文本特征，并进行归一化。
3. 使用 CLIP 图像编码器提取输入图像特征，并进行归一化。
4. 将 known 与 unknown 的文本特征合并为统一的 `text_features`。
5. 计算 `image_features @ text_features.T` 得到图文相似度矩阵。
6. 对每张图像取相似度最大的候选标签作为预测结果。

本方法本质上是 10 类候选标签上的 zero-shot 分类，避免了阈值来源不明确的问题。

## 4. 项目文件说明

```text
Advanced_Deep_Learning_first_scheme/
├── main.py              # 主程序：模型加载、文本特征计算、统一 10 类 zero-shot 评估、结果保存
├── data_loader.py       # 本地 Food_Multimodal 与 Food-101 数据加载、英文 prompt 构造
├── arguments.py         # 命令行参数定义
├── requirements.txt     # 依赖版本
├── run.sh               # 单次实验运行脚本
├── README.md            # 实验说明文档
└── .gitignore
```

说明：由于本版本不再使用阈值拒识，因此删除了 `run_threshold_sweep.sh`。

## 5. 环境配置

建议使用独立 conda 环境：

```bash
conda create -n deep python=3.10 -y
conda activate deep
pip install -r requirements.txt
```

如果服务器需要安装 CUDA 11.8 对应的 PyTorch，也可以先安装 PyTorch，再安装其余依赖：

```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.38.2 datasets==2.18.0 numpy==1.26.4 Pillow pandas tqdm scikit-learn accelerate
```

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
  --temperature 0.07 \
  --prompt_ensemble true \
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

也可以在 `run.sh` 中修改：

```bash
export CUDA_VISIBLE_DEVICES=0
```

## 8. 输出结果

程序会在 `output_dir` 中生成：

```text
results.txt
```

其中包括：

- 实验参数配置；
- known classes 与 unknown classes 设置；
- 已知类 known-only closed-set zero-shot 诊断准确率；
- 统一 10 类 zero-shot 开放集总体准确率；
- known 样本在 10 类候选标签中的准确率；
- unknown 样本在 10 类候选标签中的准确率；
- known/unknown 区域判断准确率；
- 10 类分类报告；
- 10 类混淆矩阵；
- 预测类别分布。

## 9. 主要修改说明

相较于原阈值拒识版本，本版本做了以下修改：

1. 删除阈值相关逻辑，不再使用 `threshold` 或 `use_energy_score` 参数。
2. 删除 `run_threshold_sweep.sh`，不再进行阈值扫描。
3. 将 known 与 unknown 文本特征合并为统一的 10 类 `text_features`。
4. 在 `main.py` 中直接计算 `image_features` 与全部候选 `text_features` 的相似度，并取最大值对应类别作为预测结果。
5. `data_loader.py` 中的 `get_class_prompts` 去掉 `class_names_zh` 参数，仅使用英文 prompt。
6. README 改写为统一候选标签 zero-shot 分类方案说明。

## 10. 注意事项

- 本项目没有对目标数据进行有监督训练或微调，符合 zero-shot 要求。
- 本版本的 unknown classes 是有具体标签的候选类别，不是一个抽象的 `unknown` 拒识标签。
- 如果 HuggingFace Food-101 下载失败，可以提前在联网环境缓存数据集，或将 Food-101 子集改成本地读取。
- 如果显存不足，可以减小 `--per_device_eval_batch_size`。
