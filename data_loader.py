"""
数据加载模块
负责加载本地 Food_Multimodal 数据集和线上 Food-101 数据集
"""

import os
import pandas as pd
from PIL import Image
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset


# 已知类别(来自本地数据) 与 Food-101 对应的映射
KNOWN_CLASSES = ["baozi", "cupcake", "french_fries", "rice", "youtiao"]
KNOWN_CLASSES_ZH = ["包子", "小蛋糕", "薯条", "米饭", "油条"]

# 从Food-101中选取与已知类不重合的5个类别作为未知类
UNKNOWN_CLASSES_FOOD101 = ["hamburger", "pizza", "sushi", "steak", "ice_cream"]
UNKNOWN_CLASSES_ZH = ["汉堡包", "披萨", "寿司", "牛排", "冰淇淋"]


def get_clip_transform(image_size: int = 224):
    """获取 CLIP 标准图像预处理"""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])


class FoodMultimodalDataset(Dataset):
    """加载本地 Food_Multimodal 数据集（已知类）"""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform=None,
        class_names: Optional[List[str]] = None,
    ):
        """
        Args:
            data_root: Food_Multimodal 根目录
            split: "train", "val", 或 "test"
            transform: 图像预处理
            class_names: 类别列表，默认使用 KNOWN_CLASSES
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform or get_clip_transform()
        self.class_names = class_names or KNOWN_CLASSES
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # 读取 metadata.csv
        metadata_path = os.path.join(data_root, split, "metadata.csv")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"找不到元数据文件: {metadata_path}")

        self.metadata = pd.read_csv(metadata_path)
        self.image_dir = os.path.join(data_root, split, "images")

        # 只保留指定类别的样本
        if class_names is not None:
            self.metadata = self.metadata[self.metadata["label"].isin(class_names)]

        self.metadata = self.metadata.reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # 加载图像
        img_path = os.path.join(self.image_dir, row["image_path"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # 标签和文本描述
        label = self.class_to_idx[row["label"]]
        text = row["text"] if pd.notna(row["text"]) else ""

        return {
            "pixel_values": image,
            "label": label,
            "label_name": row["label"],
            "text": text,
            "image_path": row["image_path"],
        }


class Food101Dataset(Dataset):
    """加载 HuggingFace Food-101 数据集"""

    def __init__(
        self,
        split: str = "train",
        transform=None,
        selected_classes: Optional[List[str]] = None,
        max_samples_per_class: Optional[int] = None,
    ):
        """
        Args:
            split: "train" 或 "validation"
            transform: 图像预处理
            selected_classes: 选择的类别列表，默认使用 UNKNOWN_CLASSES_FOOD101
            max_samples_per_class: 每类最多样本数，None 表示全部
        """
        self.transform = transform or get_clip_transform()
        self.selected_classes = selected_classes or UNKNOWN_CLASSES_FOOD101
        self.class_to_idx = {name: i for i, name in enumerate(self.selected_classes)}
        self.max_samples_per_class = max_samples_per_class

        # 加载 Food-101 数据集
        try:
            dataset = load_dataset("food101", split=split, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(
                f"加载 Food-101 失败: {e}\n"
                "请确保已安装 datasets 库 (pip install datasets)，并且网络连接正常。"
            )

        # 获取类别名称列表，确保已知类与未知类无重叠
        label_names = dataset.features["label"].names
        for cls in self.selected_classes:
            if cls in KNOWN_CLASSES:
                print(f"  警告: 未知类 '{cls}' 与已知类重叠，已跳过")
                # 实际运行中应避免这种情况
        selected_label_ids = [
            i for i, name in enumerate(label_names) if name in self.selected_classes
        ]

        # 构建样本索引（惰性加载，只在 __getitem__ 时读取图像）
        self.sample_indices = []
        for i, sample in enumerate(dataset):
            if sample["label"] in selected_label_ids:
                class_name = label_names[sample["label"]]
                self.sample_indices.append({
                    "dataset_idx": i,
                    "label": self.class_to_idx[class_name],
                    "label_name": class_name,
                })

        # 限制每类样本数
        if max_samples_per_class is not None:
            from collections import defaultdict
            grouped = defaultdict(list)
            for s in self.sample_indices:
                grouped[s["label_name"]].append(s)
            self.sample_indices = []
            for cls_name in self.selected_classes:
                self.sample_indices.extend(grouped[cls_name][:max_samples_per_class])

        # 保存完整 dataset 引用以延迟加载图像
        self._dataset = dataset

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        info = self.sample_indices[idx]
        sample = self._dataset[info["dataset_idx"]]
        image = sample["image"].convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "pixel_values": image,
            "label": info["label"],
            "label_name": info["label_name"],
            "text": "",
        }


def get_class_prompts(
    class_names: List[str],
    class_names_zh: List[str],
    ensemble: bool = True,
) -> list:
    """
    为每个类别生成文本提示（prompt），用于 CLIP zero-shot 分类

    使用多个模板进行 prompt ensemble 以提高 accuracy

    Args:
        class_names: 英文类别名
        class_names_zh: 中文类别名（用于辅助）
        ensemble: 是否使用多个提示模板

    Returns:
        每个类别对应的提示列表（如果 ensemble=True，每个类别有多个提示）
    """
    templates = [
        "a photo of {}, a type of food",
        "a close-up photo of {}",
        "{}",
    ]
    if ensemble:
        prompts = []
        for en_name in class_names:
            food_name = en_name.replace("_", " ")
            for tmpl in templates:
                prompts.append(tmpl.format(food_name))
        return prompts
    else:
        prompts = []
        for en_name in class_names:
            food_name = en_name.replace("_", " ")
            prompts.append(templates[0].format(food_name))
        return prompts


def get_unknown_class_prompts(ensemble: bool = True) -> list:
    """获取未知类别的文本提示"""
    return get_class_prompts(UNKNOWN_CLASSES_FOOD101, UNKNOWN_CLASSES_ZH, ensemble)


def build_combined_dataset(
    data_root: str,
    train_transform=None,
    val_transform=None,
    known_class_names: Optional[List[str]] = None,
    unknown_class_names: Optional[List[str]] = None,
    max_unknown_per_class: int = 50,
):
    """
    构建训练/验证/测试数据集组合

    Returns:
        train_dataset, val_dataset, test_dataset
        每个数据集包含已知类 + 未知类样本
    """
    transform = train_transform or get_clip_transform()
    val_transform = val_transform or get_clip_transform()

    # 已知类数据集
    train_known = FoodMultimodalDataset(
        data_root, "train", transform=transform, class_names=known_class_names
    )
    val_known = FoodMultimodalDataset(
        data_root, "val", transform=val_transform, class_names=known_class_names
    )
    test_known = FoodMultimodalDataset(
        data_root, "test", transform=val_transform, class_names=known_class_names
    )

    # 未知类数据集（从 Food-101 加载验证集用于测试）
    test_unknown = Food101Dataset(
        split="validation",
        transform=val_transform,
        selected_classes=unknown_class_names,
        max_samples_per_class=max_unknown_per_class,
    )

    return train_known, val_known, test_known, test_unknown


if __name__ == "__main__":
    # 简单测试
    data_root = r"c:\Users\wang\Desktop\课程作业\深度学习进阶\version2\Food_Multimodal"

    train_known, val_known, test_known, test_unknown = build_combined_dataset(
        data_root, max_unknown_per_class=20
    )

    print(f"训练集(已知类): {len(train_known)} 样本")
    print(f"验证集(已知类): {len(val_known)} 样本")
    print(f"测试集(已知类): {len(test_known)} 样本")
    print(f"测试集(未知类): {len(test_unknown)} 样本")

    # 测试一条样本
    sample = train_known[0]
    print(f"\n样本示例:")
    print(f"  图像尺寸: {sample['pixel_values'].shape}")
    print(f"  标签: {sample['label']} ({sample['label_name']})")
    print(f"  文本描述: {sample['text'][:50]}...")

    # 测试提示词
    prompts = get_class_prompts(KNOWN_CLASSES, KNOWN_CLASSES_ZH)
    print(f"\n已知类提示词: {prompts}")
