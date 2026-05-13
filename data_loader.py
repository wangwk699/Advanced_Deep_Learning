"""数据加载模块。

负责加载本地 Food_Multimodal 数据集中的 5 个已知类，以及从 HuggingFace Food-101
中选取若干与已知类不重合的类别作为开放集 unknown classes。
"""

import os
from collections import defaultdict
from typing import List, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset

# 已知类别：对应大作业要求中的薯条、油条、包子、米饭、小蛋糕。
# 注意顺序会影响类别 id，请与 metadata.csv 中的 label 保持一致。
KNOWN_CLASSES = ["baozi", "cupcake", "french_fries", "rice", "youtiao"]
KNOWN_CLASSES_ZH = ["包子", "小蛋糕", "薯条", "米饭", "油条"]

# 从 Food-101 选取与上述 5 类不重合的类别作为 unknown classes。
UNKNOWN_CLASSES_FOOD101 = ["hamburger", "pizza", "sushi", "steak", "ice_cream"]
UNKNOWN_CLASSES_ZH = ["汉堡包", "披萨", "寿司", "牛排", "冰淇淋"]


def get_clip_transform(image_size: int = 224):
    """获取 CLIP 图像预处理。"""
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


class FoodMultimodalDataset(Dataset):
    """加载本地 Food_Multimodal 数据集。"""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform=None,
        class_names: Optional[List[str]] = None,
    ):
        self.data_root = data_root
        self.split = split
        self.transform = transform or get_clip_transform()
        self.class_names = class_names or KNOWN_CLASSES
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        metadata_path = os.path.join(data_root, split, "metadata.csv")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"找不到元数据文件: {metadata_path}")

        self.metadata = pd.read_csv(metadata_path)
        self.image_dir = os.path.join(data_root, split, "images")

        if class_names is not None:
            self.metadata = self.metadata[self.metadata["label"].isin(class_names)]
            self.metadata = self.metadata.reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_path"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label_name = row["label"]
        label = self.class_to_idx[label_name]
        text = row["text"] if "text" in row and pd.notna(row["text"]) else ""

        return {
            "pixel_values": image,
            "label": label,
            "label_name": label_name,
            "text": text,
            "image_path": row["image_path"],
        }


class Food101Dataset(Dataset):
    """加载 HuggingFace Food-101 数据集中的 selected_classes 作为未知类。"""

    def __init__(
        self,
        split: str = "validation",
        transform=None,
        selected_classes: Optional[List[str]] = None,
        max_samples_per_class: Optional[int] = None,
    ):
        self.transform = transform or get_clip_transform()
        self.selected_classes = selected_classes or UNKNOWN_CLASSES_FOOD101
        self.class_to_idx = {name: i for i, name in enumerate(self.selected_classes)}
        self.max_samples_per_class = max_samples_per_class

        try:
            dataset = load_dataset("food101", split=split, trust_remote_code=True)
        except Exception as exc:
            raise RuntimeError(
                f"加载 Food-101 失败: {exc}\n"
                "请确保已安装 datasets 库，并且服务器可以访问 HuggingFace 数据集。"
            ) from exc

        label_names = dataset.features["label"].names
        overlap = set(self.selected_classes) & set(KNOWN_CLASSES)
        if overlap:
            raise ValueError(f"未知类不能与已知类重叠: {sorted(overlap)}")

        selected_label_ids = [
            i for i, name in enumerate(label_names) if name in self.selected_classes
        ]
        missing = sorted(set(self.selected_classes) - set(label_names))
        if missing:
            raise ValueError(f"Food-101 中找不到这些未知类: {missing}")

        self.sample_indices = []
        for i, sample in enumerate(dataset):
            if sample["label"] in selected_label_ids:
                class_name = label_names[sample["label"]]
                self.sample_indices.append(
                    {
                        "dataset_idx": i,
                        "label": self.class_to_idx[class_name],
                        "label_name": class_name,
                    }
                )

        if max_samples_per_class is not None:
            grouped = defaultdict(list)
            for sample in self.sample_indices:
                grouped[sample["label_name"]].append(sample)

            self.sample_indices = []
            for cls_name in self.selected_classes:
                self.sample_indices.extend(grouped[cls_name][:max_samples_per_class])

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
            "image_path": f"food101:{info['dataset_idx']}",
        }


def get_class_prompts(
    class_names: List[str],
    class_names_zh: Optional[List[str]] = None,
    ensemble: bool = True,
) -> List[str]:
    """为 CLIP zero-shot 分类生成文本提示。

    返回顺序按类别分组：每个类别连续生成相同数量的 prompt，方便后续做
    prompt ensemble 平均。
    """
    templates = [
        "a photo of {}, a type of food",
        "a close-up photo of {}",
        "a plate of {}",
        "{}",
    ]

    prompts = []
    for en_name in class_names:
        food_name = en_name.replace("_", " ")
        if ensemble:
            for template in templates:
                prompts.append(template.format(food_name))
        else:
            prompts.append(templates[0].format(food_name))
    return prompts


def get_unknown_class_prompts(ensemble: bool = True) -> List[str]:
    return get_class_prompts(UNKNOWN_CLASSES_FOOD101, UNKNOWN_CLASSES_ZH, ensemble)


def build_combined_dataset(
    data_root: str,
    train_transform=None,
    val_transform=None,
    known_class_names: Optional[List[str]] = None,
    unknown_class_names: Optional[List[str]] = None,
    max_unknown_per_class: int = 50,
):
    """构建已知类训练/验证/测试集和 Food-101 未知类测试集。"""
    transform = train_transform or get_clip_transform()
    val_transform = val_transform or get_clip_transform()

    train_known = FoodMultimodalDataset(
        data_root, "train", transform=transform, class_names=known_class_names
    )
    val_known = FoodMultimodalDataset(
        data_root, "val", transform=val_transform, class_names=known_class_names
    )
    test_known = FoodMultimodalDataset(
        data_root, "test", transform=val_transform, class_names=known_class_names
    )
    test_unknown = Food101Dataset(
        split="validation",
        transform=val_transform,
        selected_classes=unknown_class_names,
        max_samples_per_class=max_unknown_per_class,
    )
    return train_known, val_known, test_known, test_unknown
