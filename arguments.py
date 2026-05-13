"""
超参数配置模块
使用 HuggingFace 的 HfArgumentParser 解析命令行参数
"""

from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments, HfArgumentParser
import argparse


@dataclass
class OurArguments(TrainingArguments):
    """
    自定义训练/评估参数，继承 HuggingFace TrainingArguments
    """

    # 数据参数
    data_root: str = field(
        default="./Food_Multimodal",
        metadata={"help": "本地 Food_Multimodal 数据集根目录路径"},
    )
    max_unknown_per_class: int = field(
        default=50,
        metadata={"help": "从 Food-101 每类选取的未知类样本数"},
    )

    # 模型参数
    model_name: str = field(
        default="openai/clip-vit-base-patch32",
        metadata={"help": "使用的 CLIP 模型名称"},
    )
    image_size: int = field(
        default=224,
        metadata={"help": "输入图像尺寸"},
    )

    # 评估参数
    threshold: float = field(
        default=0.3,
        metadata={"help": "开放集识别的相似度阈值，低于此值判定为未知类"},
    )
    use_energy_score: bool = field(
        default=False,
        metadata={"help": "是否使用能量分数代替最大相似度进行开放集检测"},
    )
    temperature: float = field(
        default=0.07,
        metadata={"help": "CLIP 的 logit_scale 温度参数"},
    )

    # 输出参数
    output_dir: str = field(
        default="./results",
        metadata={"help": "输出目录"},
    )


def parse_args() -> OurArguments:
    """
    解析命令行参数

    Usage:
        CUDA_VISIBLE_DEVICES=0 python main.py \\
            --data_root ./Food_Multimodal \\
            --model_name openai/clip-vit-base-patch32 \\
            --threshold 0.3 \\
            --output_dir ./results
    """
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print("=" * 60)
    print("实验配置:")
    print(args)
    print("=" * 60)
    return args


if __name__ == "__main__":
    # 测试参数解析
    args = parse_args()
