"""基于 CLIP 的开放集 zero-shot 食物图像识别。

流程：
1. 加载本地 Food_Multimodal 数据集中的 5 个已知类。
2. 从 Food-101 中加载与已知类不重合的类别作为 unknown classes。
3. 使用 CLIP 图文对齐能力进行 zero-shot 分类，不做有监督训练或微调。
4. 对已知类样本输出 5 类之一；对 unknown 样本先检测是否未知，再输出候选 unknown 类别。
5. 输出 closed-set 与 open-set 评估结果，并保存到 results.txt。
"""

import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from arguments import parse_args
from data_loader import (
    Food101Dataset,
    FoodMultimodalDataset,
    KNOWN_CLASSES,
    KNOWN_CLASSES_ZH,
    UNKNOWN_CLASSES_FOOD101,
    UNKNOWN_CLASSES_ZH,
    get_class_prompts,
    get_clip_transform,
)

UNKNOWN_INDEX = len(KNOWN_CLASSES)


def _encode_text_prompts(
    model: CLIPModel,
    processor: CLIPProcessor,
    prompts: List[str],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """编码文本提示，并对每个类别的多个 prompt 做平均。"""
    inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    raw_features = model.get_text_features(**inputs)
    raw_features = raw_features / raw_features.norm(dim=-1, keepdim=True)

    if len(prompts) > num_classes:
        prompts_per_class = len(prompts) // num_classes
        text_features = raw_features.view(num_classes, prompts_per_class, -1).mean(dim=1)
    else:
        text_features = raw_features

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def evaluate_known_classes(
    model: CLIPModel,
    text_features: torch.Tensor,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, torch.Tensor, torch.Tensor, List[int]]:
    """评估 5 个已知类的 closed-set zero-shot 分类准确率。"""
    model.eval()
    all_logits = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估已知类"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            image_features = model.get_image_features(pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

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
    """评估开放集识别性能。

    规则：
    - 对已知类样本：若对 5 个 known prompt 的最大置信度 > threshold，则输出已知类；否则输出 unknown。
    - 对 unknown 样本：若最大置信度 <= threshold，则判定为 unknown，并进一步在 unknown 候选标签中输出具体类别。
    """
    model.eval()

    known_preds: List[int] = []
    known_labels: List[int] = []
    known_confidences: List[float] = []
    known_is_unknown: List[bool] = []

    open_true_labels: List[int] = []
    open_pred_labels: List[int] = []

    with torch.no_grad():
        for batch in tqdm(known_dataloader, desc="开放集 - 已知类"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            image_features = model.get_image_features(pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = model.logit_scale.exp() * image_features @ known_text_features.T

            if use_energy:
                confidence = torch.logsumexp(logits, dim=-1)
            else:
                confidence, _ = logits.max(dim=-1)

            is_unknown = confidence <= threshold
            preds = logits.argmax(dim=-1)
            preds_for_open_report = preds.clone()
            preds_for_open_report[is_unknown] = UNKNOWN_INDEX

            known_preds.extend(preds.cpu().tolist())
            known_labels.extend(labels.cpu().tolist())
            known_confidences.extend(confidence.cpu().tolist())
            known_is_unknown.extend(is_unknown.cpu().tolist())

            open_true_labels.extend(labels.cpu().tolist())
            open_pred_labels.extend(preds_for_open_report.cpu().tolist())

    unknown_preds: List[int] = []
    unknown_labels: List[int] = []
    unknown_confidences: List[float] = []
    unknown_is_detected: List[bool] = []
    unknown_final_names: List[str] = []

    with torch.no_grad():
        for batch in tqdm(unknown_dataloader, desc="开放集 - 未知类"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            image_features = model.get_image_features(pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            known_logits = model.logit_scale.exp() * image_features @ known_text_features.T
            if use_energy:
                confidence = torch.logsumexp(known_logits, dim=-1)
            else:
                confidence, _ = known_logits.max(dim=-1)

            detected = confidence <= threshold

            # 对所有 unknown 样本都计算其在 unknown 候选标签中的具体预测类别。
            unknown_logits = model.logit_scale.exp() * image_features @ unknown_text_features.T
            unknown_pred = unknown_logits.argmax(dim=-1)

            unknown_preds.extend(unknown_pred.cpu().tolist())
            unknown_labels.extend(labels.cpu().tolist())
            unknown_confidences.extend(confidence.cpu().tolist())
            unknown_is_detected.extend(detected.cpu().tolist())
            unknown_final_names.extend(
                [UNKNOWN_CLASSES_FOOD101[p] for p in unknown_pred.cpu().tolist()]
            )

            # 开放集总报告只评估 known-vs-unknown 检测：unknown 统一映射为 UNKNOWN_INDEX。
            open_true_labels.extend([UNKNOWN_INDEX] * labels.size(0))
            known_side_pred = known_logits.argmax(dim=-1)
            preds_for_open_report = known_side_pred.clone()
            preds_for_open_report[detected] = UNKNOWN_INDEX
            open_pred_labels.extend(preds_for_open_report.cpu().tolist())

    known_total = len(known_labels)
    unknown_total = len(unknown_labels)
    known_detected_as_known = known_total - sum(known_is_unknown)
    unknown_detected_as_unknown = sum(unknown_is_detected)

    known_correct = sum(
        1 for pred, label, rejected in zip(known_preds, known_labels, known_is_unknown)
        if (not rejected) and pred == label
    )
    known_accuracy_open = known_correct / known_total if known_total else 0.0

    # unknown 具体类别准确率：只有被判定为 unknown 且候选 unknown 类预测正确才算正确。
    unknown_class_correct = sum(
        1
        for pred, label, detected in zip(unknown_preds, unknown_labels, unknown_is_detected)
        if detected and pred == label
    )
    unknown_class_accuracy = unknown_class_correct / unknown_total if unknown_total else 0.0

    known_recall = known_detected_as_known / known_total if known_total else 0.0
    unknown_detection_rate = unknown_detected_as_unknown / unknown_total if unknown_total else 0.0
    open_set_detection_accuracy = (
        (known_detected_as_known + unknown_detected_as_unknown) / (known_total + unknown_total)
        if (known_total + unknown_total)
        else 0.0
    )

    return {
        "known_total": known_total,
        "unknown_total": unknown_total,
        "known_accuracy_open": known_accuracy_open,
        "known_recall": known_recall,
        "unknown_detection_rate": unknown_detection_rate,
        "unknown_class_accuracy": unknown_class_accuracy,
        "unknown_class_correct": int(unknown_class_correct),
        "open_set_detection_accuracy": open_set_detection_accuracy,
        "known_detected_as_known": int(known_detected_as_known),
        "unknown_detected_as_unknown": int(unknown_detected_as_unknown),
        "known_misclassified_as_unknown": int(known_total - known_detected_as_known),
        "unknown_misclassified_as_known": int(unknown_total - unknown_detected_as_unknown),
        "known_pred_distribution": dict(Counter(known_preds)),
        "unknown_pred_distribution": dict(Counter(unknown_final_names)),
        "open_true_labels": open_true_labels,
        "open_pred_labels": open_pred_labels,
    }


def _write_dict_results(file_obj, results: Dict):
    for key, value in results.items():
        if key in {"open_true_labels", "open_pred_labels"}:
            continue
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

    with torch.no_grad():
        model.logit_scale.data = torch.clamp(
            torch.tensor([np.log(1 / args.temperature)], device=device),
            min=np.log(1 / 100),
            max=np.log(100),
        )

    print("\n" + "=" * 60)
    print("3. 构建文本提示并计算文本特征")
    known_prompts = get_class_prompts(KNOWN_CLASSES, KNOWN_CLASSES_ZH)
    unknown_prompts = get_class_prompts(UNKNOWN_CLASSES_FOOD101, UNKNOWN_CLASSES_ZH)
    print(f"已知类: {KNOWN_CLASSES}")
    print(f"未知类: {UNKNOWN_CLASSES_FOOD101}")
    print(f"已知类 prompt 数: {len(known_prompts)}")
    print(f"未知类 prompt 数: {len(unknown_prompts)}")

    with torch.no_grad():
        known_text_features = _encode_text_prompts(
            model, processor, known_prompts, len(KNOWN_CLASSES), device
        )
        unknown_text_features = _encode_text_prompts(
            model, processor, unknown_prompts, len(UNKNOWN_CLASSES_FOOD101), device
        )

    batch_size = args.per_device_eval_batch_size or 32
    num_workers = args.dataloader_num_workers if args.dataloader_num_workers is not None else 2

    val_loader = DataLoader(val_known, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_known, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    unknown_loader = DataLoader(test_unknown, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("\n" + "=" * 60)
    print("4. 已知类 closed-set zero-shot 分类评估")
    val_acc, _, val_labels, val_preds = evaluate_known_classes(
        model, known_text_features, val_loader, device
    )
    test_acc, _, test_labels, test_preds = evaluate_known_classes(
        model, known_text_features, test_loader, device
    )
    print(f"验证集 Top-1 准确率: {val_acc * 100:.2f}%")
    print(f"测试集 Top-1 准确率: {test_acc * 100:.2f}%")

    print("\n===== 已知类 closed-set 分类报告 =====")
    closed_report = classification_report(
        test_labels,
        test_preds,
        labels=list(range(len(KNOWN_CLASSES))),
        target_names=KNOWN_CLASSES,
        zero_division=0,
    )
    print(closed_report)

    print("\n" + "=" * 60)
    print(f"5. 开放集识别评估 (threshold={args.threshold})")
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

    print("\n===== 开放集识别结果 =====")
    for key, value in open_set_results.items():
        if key in {"open_true_labels", "open_pred_labels"}:
            continue
        if isinstance(value, float):
            print(f"{key}: {value * 100:.2f}%")
        else:
            print(f"{key}: {value}")

    print("\n===== 开放集 known/unknown 分类报告 =====")
    open_target_names = KNOWN_CLASSES + ["unknown"]
    open_report = classification_report(
        open_set_results["open_true_labels"],
        open_set_results["open_pred_labels"],
        labels=list(range(len(open_target_names))),
        target_names=open_target_names,
        zero_division=0,
    )
    print(open_report)

    print("\n===== 开放集混淆矩阵 =====")
    open_cm = confusion_matrix(
        open_set_results["open_true_labels"],
        open_set_results["open_pred_labels"],
        labels=list(range(len(open_target_names))),
    )
    print(open_cm)

    print("\n" + "=" * 60)
    print("6. 保存结果")
    result_path = os.path.join(args.output_dir, "results.txt")
    with open(result_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("===== 实验配置 =====\n")
        file_obj.write(str(args) + "\n\n")

        file_obj.write("===== 类别设置 =====\n")
        file_obj.write(f"已知类: {KNOWN_CLASSES}\n")
        file_obj.write(f"已知类中文: {KNOWN_CLASSES_ZH}\n")
        file_obj.write(f"未知类: {UNKNOWN_CLASSES_FOOD101}\n")
        file_obj.write(f"未知类中文: {UNKNOWN_CLASSES_ZH}\n\n")

        file_obj.write("===== Zero-shot 已知类 closed-set 评估 =====\n")
        file_obj.write(f"验证集准确率: {val_acc * 100:.2f}%\n")
        file_obj.write(f"测试集准确率: {test_acc * 100:.2f}%\n")
        file_obj.write(closed_report + "\n")

        file_obj.write("===== 开放集识别结果 =====\n")
        file_obj.write(f"阈值: {args.threshold}\n")
        file_obj.write(f"使用 energy score: {args.use_energy_score}\n")
        _write_dict_results(file_obj, open_set_results)

        file_obj.write("\n===== 开放集 known/unknown 分类报告 =====\n")
        file_obj.write(open_report + "\n")
        file_obj.write("\n===== 开放集混淆矩阵 =====\n")
        file_obj.write(str(open_cm) + "\n")

    print(f"结果已保存到: {result_path}")
    print("实验完成！")


if __name__ == "__main__":
    main()
