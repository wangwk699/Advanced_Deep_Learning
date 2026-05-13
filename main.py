"""
主函数 - 基于 CLIP 的开放集 Zero-shot 食物图像识别

工作流程：
1. 加载本地 Food_Multimodal 数据集（5个已知类）
2. 从 Food-101 加载不重合的类别作为未知类
3. 使用 CLIP 进行 zero-shot 分类
4. 实现开放集识别：已知类准确率 + 未知类检测
5. 输出评估结果
"""

import os
import sys
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from transformers import CLIPModel, CLIPProcessor

from data_loader import (
    FoodMultimodalDataset,
    Food101Dataset,
    KNOWN_CLASSES,
    KNOWN_CLASSES_ZH,
    UNKNOWN_CLASSES_FOOD101,
    UNKNOWN_CLASSES_ZH,
    get_class_prompts,
    get_clip_transform,
)
from arguments import parse_args


def evaluate_known_classes(
    model: CLIPModel,
    text_features: torch.Tensor,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, torch.Tensor, torch.Tensor, List[int]]:
    """
    评估已知类分类准确率

    Returns:
        accuracy, all_logits, all_labels, all_preds
    """
    model.eval()
    all_logits = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估已知类"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            # 提取图像特征
            image_features = model.get_image_features(pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 计算相似度
            logits = model.logit_scale.exp() * image_features @ text_features.T

            preds = logits.argmax(dim=-1)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)

    accuracy = (all_preds == all_labels).float().mean().item()

    return accuracy, all_logits, all_labels, all_preds.tolist()


def evaluate_open_set(
    model: CLIPModel,
    known_text_features: torch.Tensor,
    unknown_text_features: torch.Tensor,
    known_dataloader: DataLoader,
    unknown_dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.3,
    use_energy: bool = False,
) -> Dict:
    """
    评估开放集识别性能

    策略：
    - 计算图像与已知类文本的相似度
    - 如果 confidence > threshold，判定为已知类（并输出对应类别）
    - 如果 confidence <= threshold，判定为未知类
    - 对于判定为未知类的样本，计算与未知类候选标签的相似度

    Args:
        threshold: 置信度阈值
        use_energy: 是否使用能量分数代替最大相似度

    Returns:
        包含各项指标的字典
    """
    model.eval()

    # --- 已知类评估（开放集场景）---
    known_preds = []
    known_labels = []
    known_confidences = []  # 置信度分数
    known_is_unknown = []   # 被判定为未知的已知类样本

    with torch.no_grad():
        for batch in tqdm(known_dataloader, desc="开放集 - 已知类"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            image_features = model.get_image_features(pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = model.logit_scale.exp() * image_features @ known_text_features.T

            # 计算置信度
            if use_energy:
                confidence = torch.logsumexp(logits, dim=-1)
            else:
                confidence, _ = logits.max(dim=-1)

            # 判定：confidence > threshold → 已知类，否则未知类
            is_unknown = confidence <= threshold
            preds = logits.argmax(dim=-1)
            preds[is_unknown] = -1  # -1 表示未知类

            known_preds.extend(preds.cpu().tolist())
            known_labels.extend(labels.cpu().tolist())
            known_confidences.extend(confidence.cpu().tolist())
            known_is_unknown.extend(is_unknown.cpu().tolist())

    # --- 未知类评估 ---
    unknown_preds = []
    unknown_labels = []
    unknown_confidences = []
    unknown_is_detected = []  # 被正确检测为未知的样本

    with torch.no_grad():
        for batch in tqdm(unknown_dataloader, desc="开放集 - 未知类"):
            pixel_values = batch["pixel_values"].to(device)

            image_features = model.get_image_features(pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = model.logit_scale.exp() * image_features @ known_text_features.T

            # 计算置信度
            if use_energy:
                confidence = torch.logsumexp(logits, dim=-1)
            else:
                confidence, _ = logits.max(dim=-1)

            # 未知类检测：confidence <= threshold 时正确检测
            detected = (confidence <= threshold).float()

            # 被检测为"已知"的未知类样本，用未知类候选标签预测
            unknown_logits = model.logit_scale.exp() * image_features @ unknown_text_features.T
            unknown_pred = unknown_logits.argmax(dim=-1)

            unknown_preds.extend(unknown_pred.cpu().tolist())
            unknown_labels.extend(batch["label"].cpu().tolist())
            unknown_confidences.extend(confidence.cpu().tolist())
            unknown_is_detected.extend(detected.cpu().tolist())

    # --- 计算指标 ---
    known_total = len(known_labels)
    unknown_total = len(unknown_labels)
    known_detected_as_known = known_total - sum(known_is_unknown)
    unknown_detected_as_unknown = sum(unknown_is_detected)

    # 已知类准确率（仅对判定为已知的样本）
    known_correct = sum(
        1 for p, l in zip(known_preds, known_labels) if p == l
    )
    known_accuracy_closed = known_correct / known_total if known_total > 0 else 0

    # 开放集指标
    known_recall = known_detected_as_known / known_total if known_total > 0 else 0.0
    unknown_recall = unknown_detected_as_unknown / unknown_total if unknown_total > 0 else 0.0
    open_set_accuracy = (known_detected_as_known + unknown_detected_as_unknown) / (
        known_total + unknown_total
    ) if (known_total + unknown_total) > 0 else 0.0

    # 已知类预测分布
    from collections import Counter
    known_pred_dist = Counter(known_preds)

    return {
        "known_total": known_total,
        "unknown_total": unknown_total,
        "known_accuracy": known_accuracy_closed,
        "known_recall": known_recall,             # 已知类被正确保留的比例
        "unknown_detection_rate": unknown_recall,  # 未知类被正确检测的比例
        "open_set_accuracy": open_set_accuracy,
        "known_detected_as_known": int(known_detected_as_known),
        "unknown_detected_as_unknown": int(unknown_detected_as_unknown),
        "known_misclassified_as_unknown": int(known_total - known_detected_as_known),
        "unknown_misclassified_as_known": int(unknown_total - unknown_detected_as_unknown),
        "known_pred_distribution": dict(known_pred_dist),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ==================== 1. 加载数据 ====================
    print("\n" + "=" * 60)
    print("加载数据集...")

    transform = get_clip_transform(args.image_size)

    # 已知类
    train_known = FoodMultimodalDataset(
        args.data_root, "train", transform=transform, class_names=KNOWN_CLASSES
    )
    val_known = FoodMultimodalDataset(
        args.data_root, "val", transform=transform, class_names=KNOWN_CLASSES
    )
    test_known = FoodMultimodalDataset(
        args.data_root, "test", transform=transform, class_names=KNOWN_CLASSES
    )

    # 未知类（从 Food-101 加载验证集）
    test_unknown = Food101Dataset(
        split="validation",
        transform=transform,
        selected_classes=UNKNOWN_CLASSES_FOOD101,
        max_samples_per_class=args.max_unknown_per_class,
    )

    print(f"  训练集(已知类): {len(train_known)} 张图像")
    print(f"  验证集(已知类): {len(val_known)} 张图像")
    print(f"  测试集(已知类): {len(test_known)} 张图像")
    print(f"  未知类(来自Food-101): {len(test_unknown)} 张图像")

    # ==================== 2. 加载 CLIP 模型 ====================
    print("\n" + "=" * 60)
    print(f"加载 CLIP 模型: {args.model_name}")

    model = CLIPModel.from_pretrained(args.model_name)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model = model.to(device)
    model.eval()
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 调整 logit_scale 温度
    with torch.no_grad():
        model.logit_scale.data = torch.clamp(
            torch.tensor([np.log(1 / args.temperature)]).to(device),
            min=np.log(1 / 100),
            max=np.log(100),
        )

    # ==================== 3. 构建文本提示并计算文本特征 ====================
    print("\n" + "=" * 60)
    print("构建文本提示...")

    # 已知类提示
    known_prompts = get_class_prompts(KNOWN_CLASSES, KNOWN_CLASSES_ZH)
    print(f"  已知类: {KNOWN_CLASSES}")
    print(f"  已知类提示数: {len(known_prompts)}")

    # 未知类提示
    unknown_prompts = get_class_prompts(
        UNKNOWN_CLASSES_FOOD101, UNKNOWN_CLASSES_ZH
    )
    print(f"  未知类: {UNKNOWN_CLASSES_FOOD101}")
    print(f"  未知类提示数: {len(unknown_prompts)}")

    # 计算文本特征
    print("计算文本特征...")
    with torch.no_grad():
        # 已知类文本特征
        known_inputs = processor(
            text=known_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        known_raw_features = model.get_text_features(**known_inputs)
        known_raw_features = known_raw_features / known_raw_features.norm(dim=-1, keepdim=True)

        # 如果有 ensemble 提示(每个类有多个模板)，按类别进行平均
        num_known = len(KNOWN_CLASSES)
        if len(known_prompts) > num_known:
            # 假设提示按类别分组排列
            prompts_per_class = len(known_prompts) // num_known
            known_text_features = known_raw_features.view(
                num_known, prompts_per_class, -1
            ).mean(dim=1)
        else:
            known_text_features = known_raw_features

        # 已知类特征归一化
        known_text_features = known_text_features / known_text_features.norm(dim=-1, keepdim=True)

        # 未知类文本特征
        unknown_inputs = processor(
            text=unknown_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        unknown_raw_features = model.get_text_features(**unknown_inputs)
        unknown_raw_features = unknown_raw_features / unknown_raw_features.norm(dim=-1, keepdim=True)

        num_unknown = len(UNKNOWN_CLASSES_FOOD101)
        if len(unknown_prompts) > num_unknown:
            prompts_per_class = len(unknown_prompts) // num_unknown
            unknown_text_features = unknown_raw_features.view(
                num_unknown, prompts_per_class, -1
            ).mean(dim=1)
        else:
            unknown_text_features = unknown_raw_features

        unknown_text_features = unknown_text_features / unknown_text_features.norm(dim=-1, keepdim=True)

    # ==================== 4. 已知类 Zero-shot 评估 ====================
    print("\n" + "=" * 60)
    print("4. 已知类 Zero-shot 分类评估")

    val_loader = DataLoader(
        val_known, batch_size=args.per_device_eval_batch_size or 32,
        shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_known, batch_size=args.per_device_eval_batch_size or 32,
        shuffle=False, num_workers=2
    )

    val_acc, val_logits, val_labels, val_preds = evaluate_known_classes(
        model, known_text_features, val_loader, device
    )
    test_acc, test_logits, test_labels, test_preds = evaluate_known_classes(
        model, known_text_features, test_loader, device
    )

    print(f"  验证集准确率 (Top-1): {val_acc * 100:.2f}%")
    print(f"  测试集准确率 (Top-1): {test_acc * 100:.2f}%")

    # ==================== 5. 开放集评估 ====================
    print("\n" + "=" * 60)
    print(f"5. 开放集识别评估 (threshold={args.threshold})")

    unknown_loader = DataLoader(
        test_unknown, batch_size=args.per_device_eval_batch_size or 32,
        shuffle=False, num_workers=2
    )

    open_set_results = evaluate_open_set(
        model,
        known_text_features,
        unknown_text_features,
        test_loader,
        unknown_loader,
        device,
        threshold=args.threshold,
        use_energy=args.use_energy_score,
    )

    print(f"\n===== 开放集识别结果 =====")
    print(f"  已知类样本数: {open_set_results['known_total']}")
    print(f"  未知类样本数: {open_set_results['unknown_total']}")
    print(f"  已知类 Top-1 准确率: {open_set_results['known_accuracy'] * 100:.2f}%")
    print(f"  已知类召回率: {open_set_results['known_recall'] * 100:.2f}%")
    print(f"  未知类检测率: {open_set_results['unknown_detection_rate'] * 100:.2f}%")
    print(f"  开放集总体准确率: {open_set_results['open_set_accuracy'] * 100:.2f}%")
    print(f"  已知类被正确保留: {open_set_results['known_detected_as_known']}")
    print(f"  未知类被正确检测: {open_set_results['unknown_detected_as_unknown']}")
    print(f"  已知类被误判为未知: {open_set_results['known_misclassified_as_unknown']}")
    print(f"  未知类被误判为已知: {open_set_results['unknown_misclassified_as_known']}")

    # 按类别打印已知类预测分布
    print("\n===== 已知类混淆矩阵 =====")
    from sklearn.metrics import confusion_matrix, classification_report
    full_labels = [l for l in test_labels]
    full_preds = [p if p >= 0 else len(KNOWN_CLASSES) for p in test_preds]
    target_names = KNOWN_CLASSES + ["unknown"]
    print(classification_report(
        full_labels, full_preds,
        target_names=target_names,
        zero_division=0,
    ))

    # ==================== 6. 保存结果 ====================
    print("\n" + "=" * 60)
    print("保存结果...")

    with open(os.path.join(args.output_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write("===== 实验配置 =====\n")
        f.write(str(args) + "\n\n")

        f.write("===== Zero-shot 已知类评估 =====\n")
        f.write(f"验证集准确率: {val_acc * 100:.2f}%\n")
        f.write(f"测试集准确率: {test_acc * 100:.2f}%\n\n")

        f.write("===== 开放集识别结果 =====\n")
        f.write(f"阈值: {args.threshold}\n")
        for k, v in open_set_results.items():
            if isinstance(v, float):
                f.write(f"  {k}: {v * 100:.2f}%\n")
            else:
                f.write(f"  {k}: {v}\n")

    print(f"  结果已保存到: {os.path.join(args.output_dir, 'results.txt')}")
    print("\n实验完成!")


if __name__ == "__main__":
    main()
