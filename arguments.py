"""
超参数配置模块

使用 HuggingFace 的 HfArgumentParser 解析命令行参数。
本版本采用“统一候选标签”的 zero-shot 分类方案，不再使用开放集阈值扫描。
"""

from dataclasses import dataclass, field

from transformers import HfArgumentParser, TrainingArguments


@dataclass
class OurArguments(TrainingArguments):
    """自定义训练/评估参数，继承 HuggingFace TrainingArguments。"""

    # 数据参数
    data_root: str = field(
        default="./Food_Multimodal",
        metadata={"help": "本地 Food_Multimodal 数据集根目录路径"},
    )
    max_unknown_per_class: int = field(
        default=50,
        metadata={"help": "从 Food-101 每个 unknown 类别选取的测试样本数"},
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
    temperature: float = field(
        default=0.07,
        metadata={"help": "用于重新设置 CLIP logit_scale 的温度；设为 0 或负数则保持模型默认值"},
    )
    prompt_ensemble: bool = field(
        default=True,
        metadata={"help": "是否为每个类别使用多个英文 prompt 并做特征平均"},
    )

    # 输出参数
    output_dir: str = field(
        default="./results",
        metadata={"help": "输出目录"},
    )

    templates_index : int = field(
        default=0,
        metadata={"help": "采用提示词模版下标"},
    )


def parse_args() -> OurArguments:
    """
    解析命令行参数。

    示例：
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --data_root ./Food_Multimodal \
        --model_name openai/clip-vit-base-patch32 \
        --max_unknown_per_class 50 \
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
    parse_args()
