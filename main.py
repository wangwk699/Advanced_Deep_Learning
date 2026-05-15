"""基于 CLIP 的统一候选标签 zero-shot 食物图像识别。

流程：
1. 加载本地 Food_Multimodal 数据集中的 5 个已知类。
2. 从 Food-101 中加载与已知类不重合的 5 个类别作为 unknown classes。
3. 将 known classes 与 unknown classes 合并为一个统一的 10 类候选标签集合。
4. 使用 CLIP 图文对齐能力计算 image_features 与 text_features 的相似度。
5. 对每张图像直接在 10 个候选标签中选择相似度最高者作为预测类别。

本版本不做有监督训练或微调，也不再使用阈值拒识或阈值扫描。
评价指标保留 Top-1 accuracy 与预测类别分布，不再输出 classification_report 和 confusion_matrix。
"""

import os
from collections import Counter
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from arguments import parse_args
from data_loader import (
    Food101Dataset,
    FoodMultimodalDataset,
    KNOWN_CLASSES,
    UNKNOWN_CLASSES_FOOD101,
    get_class_prompts,
    get_clip_transform,
)


def _encode_text_prompts(
    model: CLIPModel,
    processor: CLIPProcessor,
    prompts: List[str],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """编码文本提示，并对每个类别的多个 prompt 做平均。"""
    inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    raw_features = model.get_text_features(**inputs)
    raw_features = raw_features / raw_features.norm(dim=-1, keepdim=True)

    if len(prompts) > num_classes:
        if len(prompts) % num_classes != 0:
            raise ValueError(
                f"prompt 数量 {len(prompts)} 不能被类别数 {num_classes} 整除，"
                "无法进行 prompt ensemble。"
            )
        prompts_per_class = len(prompts) // num_classes
        text_features = raw_features.view(num_classes, prompts_per_class, -1).mean(dim=1)
    else:
        text_features = raw_features

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def _predict_with_text_features(
    model: CLIPModel,
    pixel_values: torch.Tensor,
    text_features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算图像与文本特征的相似度，并返回 logits 与 top-1 预测。"""
    image_features = model.get_image_features(pixel_values=pixel_values)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    logits = model.logit_scale.exp() * image_features @ text_features.T
    preds = logits.argmax(dim=-1)
    return logits, preds


def evaluate_zero_shot(
    model: CLIPModel,
    text_features: torch.Tensor,
    dataloader: DataLoader,
    device: torch.device,
    label_offset: int = 0,
    desc: str = "评估",
) -> Dict:
    """在给定候选文本特征上进行 zero-shot 分类评估。

    label_offset 用于将 Food-101 unknown 数据集中的局部标签 0..4 映射到统一标签
    空间中的 5..9。
    """
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            pixel_values = batch["pixel_values"].to(device).float()
            labels = batch["label"].to(device) + label_offset

            _, preds = _predict_with_text_features(model, pixel_values, text_features)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    preds_tensor = torch.tensor(all_preds, dtype=torch.long)
    accuracy = (preds_tensor == labels_tensor).float().mean().item() if all_labels else 0.0

    return {
        "accuracy": accuracy,
        "labels": all_labels,
        "preds": all_preds,
    }


def evaluate_unified_open_set(
    known_results: Dict,
    unknown_results: Dict,
    all_class_names: List[str],
) -> Dict:
    """汇总统一 10 类候选标签 zero-shot 评估结果。"""
    all_true = known_results["labels"] + unknown_results["labels"]
    all_pred = known_results["preds"] + unknown_results["preds"]

    overall_accuracy = (
        sum(int(t == p) for t, p in zip(all_true, all_pred)) / len(all_true)
        if all_true
        else 0.0
    )

    pred_distribution = Counter(all_class_names[p] for p in all_pred)
    unknown_pred_distribution = Counter(all_class_names[p] for p in unknown_results["preds"])

    return {
        "overall_accuracy": overall_accuracy,
        "known_accuracy": known_results["accuracy"],
        "unknown_accuracy": unknown_results["accuracy"],
        "known_total": len(known_results["labels"]),
        "unknown_total": len(unknown_results["labels"]),
        "pred_distribution": dict(pred_distribution),
        "unknown_pred_distribution": dict(unknown_pred_distribution),
    }


def _write_dict_results(file_obj, results: Dict):
    """将结果字典写入文件。"""
    for key, value in results.items():
        if isinstance(value, float):
            file_obj.write(f"{key}: {value * 100:.2f}%\n")
        else:
            file_obj.write(f"{key}: {value}\n")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("1. 加载数据集")
    transform = get_clip_transform(args.image_size)

    train_known = FoodMultimodalDataset(
        args.data_root, "train", transform=transform, class_names=KNOWN_CLASSES
    )
    val_known = FoodMultimodalDataset(
        args.data_root, "val", transform=transform, class_names=KNOWN_CLASSES
    )
    test_known = FoodMultimodalDataset(
        args.data_root, "test", transform=transform, class_names=KNOWN_CLASSES
    )
    test_unknown = Food101Dataset(
        split="validation",
        transform=transform,
        selected_classes=UNKNOWN_CLASSES_FOOD101,
        max_samples_per_class=args.max_unknown_per_class,
    )

    print(f"训练集(已知类): {len(train_known)} 张图像")
    print(f"验证集(已知类): {len(val_known)} 张图像")
    print(f"测试集(已知类): {len(test_known)} 张图像")
    print(f"未知类(来自 Food-101): {len(test_unknown)} 张图像")

    print("\n" + "=" * 60)
    print(f"2. 加载 CLIP 模型: {args.model_name}")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("保持 CLIP 预训练模型默认 logit_scale")

    print("\n" + "=" * 60)
    print("3. 构建统一候选标签，并计算文本特征")
    all_class_names = KNOWN_CLASSES + UNKNOWN_CLASSES_FOOD101
    all_prompts = get_class_prompts(all_class_names, ensemble=args.prompt_ensemble, index=args.templates_index)

    print(f"已知类 known classes: {KNOWN_CLASSES}")
    print(f"未知类 unknown classes: {UNKNOWN_CLASSES_FOOD101}")
    print(f"统一候选标签数: {len(all_class_names)}")
    print(f"prompt 数: {len(all_prompts)}")

    with torch.no_grad():
        all_text_features = _encode_text_prompts(
            model, processor, all_prompts, len(all_class_names), device
        )

    known_text_features = all_text_features[: len(KNOWN_CLASSES)]

    batch_size = args.per_device_eval_batch_size or 32
    num_workers = args.dataloader_num_workers if args.dataloader_num_workers is not None else 2

    test_loader = DataLoader(
        test_known, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    unknown_loader = DataLoader(
        test_unknown, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print("\n" + "=" * 60)
    print("4. 已知类 known-only closed-set zero-shot 诊断评估")
    test_known_closed = evaluate_zero_shot(
        model,
        known_text_features,
        test_loader,
        device,
        label_offset=0,
        desc="测试集已知类",
    )

    print(f"测试集 known-only Top-1 准确率: {test_known_closed['accuracy'] * 100:.2f}%")

    print("\n" + "=" * 60)
    print("5. 统一 10 类候选标签 zero-shot 开放集评估")
    known_open = evaluate_zero_shot(
        model,
        all_text_features,
        test_loader,
        device,
        label_offset=0,
        desc="开放集候选标签 - 已知类样本",
    )
    unknown_open = evaluate_zero_shot(
        model,
        all_text_features,
        unknown_loader,
        device,
        label_offset=len(KNOWN_CLASSES),
        desc="开放集候选标签 - 未知类样本",
    )

    open_set_results = evaluate_unified_open_set(
        known_open, unknown_open, all_class_names
    )

    print("\n===== 统一 10 类 zero-shot 开放集结果 =====")
    print(f"open_set_overall_top1_accuracy: {open_set_results['overall_accuracy'] * 100:.2f}%")
    print(f"open_set_known_accuracy: {open_set_results['known_accuracy'] * 100:.2f}%")
    print(f"open_set_unknown_accuracy: {open_set_results['unknown_accuracy'] * 100:.2f}%")
    print(f"known_total: {open_set_results['known_total']}")
    print(f"unknown_total: {open_set_results['unknown_total']}")
    print(f"pred_distribution: {open_set_results['pred_distribution']}")
    print(f"unknown_pred_distribution: {open_set_results['unknown_pred_distribution']}")

    print("\n" + "=" * 60)
    print("6. 保存结果")
    result_path = os.path.join(args.output_dir, "results.txt")
    with open(result_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("===== 实验配置 =====\n")
        file_obj.write(str(args) + "\n\n")

        file_obj.write("===== 类别设置 =====\n")
        file_obj.write(f"known classes: {KNOWN_CLASSES}\n")
        file_obj.write(f"unknown classes: {UNKNOWN_CLASSES_FOOD101}\n")
        file_obj.write(f"all candidate classes: {all_class_names}\n\n")

        file_obj.write("===== 数据集统计 =====\n")
        file_obj.write(f"训练集 known: {len(train_known)}\n")
        file_obj.write(f"验证集 known: {len(val_known)}\n")
        file_obj.write(f"测试集 known: {len(test_known)}\n")
        file_obj.write(f"测试集 unknown: {len(test_unknown)}\n\n")

        file_obj.write("===== 方法说明 =====\n")
        file_obj.write(
            "本实验将 known classes 与 unknown classes 合并为统一候选标签集合，"
            "计算每张图像与所有候选文本标签的 CLIP 相似度，并直接选择相似度最高的类别。"
            "实验过程中不进行有监督训练或微调，也不使用阈值拒识。\n\n"
        )

        file_obj.write("===== 评价结果 =====\n")
        file_obj.write(
            f"Known-only Top-1 Accuracy: {test_known_closed['accuracy'] * 100:.2f}%\n"
        )
        file_obj.write(
            f"Open-set Overall Top-1 Accuracy: "
            f"{open_set_results['overall_accuracy'] * 100:.2f}%\n"
        )
        file_obj.write(
            f"Open-set Known Accuracy: {open_set_results['known_accuracy'] * 100:.2f}%\n"
        )
        file_obj.write(
            f"Open-set Unknown Accuracy: {open_set_results['unknown_accuracy'] * 100:.2f}%\n"
        )
        file_obj.write(f"Known Test Samples: {open_set_results['known_total']}\n")
        file_obj.write(f"Unknown Test Samples: {open_set_results['unknown_total']}\n")
        file_obj.write(
            f"Prediction Distribution: {open_set_results['pred_distribution']}\n"
        )
        file_obj.write(
            f"Unknown Prediction Distribution: "
            f"{open_set_results['unknown_pred_distribution']}\n"
        )

    print(f"结果已保存到: {result_path}")
    print("实验完成！")


if __name__ == "__main__":
    main()
