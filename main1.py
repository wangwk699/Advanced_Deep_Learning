"""基于 CLIP 的统一候选标签 zero-shot 食物图像识别。

流程：
1. 加载本地 Food_Multimodal 数据集中的 5 个已知类。
2. 从 Food-101 中加载与已知类不重合的 5 个类别作为 unknown classes。
3. 将 known classes 与 unknown classes 合并为一个统一的 10 类候选标签集合。
4. 使用 CLIP 图文对齐能力计算 image_features 与 text_features 的相似度。
5. 对每张图像直接在 10 个候选标签中选择相似度最高者作为预测类别。

本版本不做有监督训练或微调，也不再使用阈值拒识或阈值扫描。
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
    all_logits: List[torch.Tensor] = []
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_label_names: List[str] = []
    all_pred_scores: List[float] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            pixel_values = batch["pixel_values"].to(device).float()
            labels = batch["label"].to(device) + label_offset

            logits, preds = _predict_with_text_features(model, pixel_values, text_features)
            scores = logits.max(dim=-1).values

            all_logits.append(logits.cpu())
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_pred_scores.extend(scores.cpu().tolist())
            all_label_names.extend(batch["label_name"])

    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    preds_tensor = torch.tensor(all_preds, dtype=torch.long)
    accuracy = (preds_tensor == labels_tensor).float().mean().item() if all_labels else 0.0

    return {
        "accuracy": accuracy,
        "logits": torch.cat(all_logits, dim=0) if all_logits else torch.empty(0),
        "labels": all_labels,
        "preds": all_preds,
        "label_names": all_label_names,
        "pred_scores": all_pred_scores,
    }


def evaluate_unified_open_set(
    known_results: Dict,
    unknown_results: Dict,
    all_class_names: List[str],
) -> Dict:
    """汇总统一 10 类候选标签 zero-shot 评估结果。"""
    num_known = len(KNOWN_CLASSES)

    all_true = known_results["labels"] + unknown_results["labels"]
    all_pred = known_results["preds"] + unknown_results["preds"]

    overall_accuracy = (
        sum(int(t == p) for t, p in zip(all_true, all_pred)) / len(all_true)
        if all_true
        else 0.0
    )

    known_accuracy = known_results["accuracy"]
    unknown_accuracy = unknown_results["accuracy"]

    # 观察模型有没有把已知类和未知类大致区分开
    known_region_correct = sum(
        int((t < num_known and p < num_known) or (t >= num_known and p >= num_known))
        for t, p in zip(all_true, all_pred)
    )
    known_unknown_region_accuracy = known_region_correct / len(all_true) if all_true else 0.0

    pred_distribution = Counter(all_class_names[p] for p in all_pred)
    unknown_pred_distribution = Counter(all_class_names[p] for p in unknown_results["preds"])

    return {
        "overall_accuracy": overall_accuracy,
        "known_accuracy": known_accuracy,
        "unknown_accuracy": unknown_accuracy,
        "known_unknown_region_accuracy": known_unknown_region_accuracy,
        "known_total": len(known_results["labels"]),
        "unknown_total": len(unknown_results["labels"]),
        "all_true_labels": all_true,
        "all_pred_labels": all_pred,
        "pred_distribution": dict(pred_distribution),
        "unknown_pred_distribution": dict(unknown_pred_distribution),
    }


def _write_dict_results(file_obj, results: Dict):
    """将结果字典写入文件，跳过过长的逐样本标签列表。"""
    skip_keys = {"all_true_labels", "all_pred_labels"}
    for key, value in results.items():
        if key in skip_keys:
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

    print(f"训练集(已知类): {len(train_known)} 张图像")             # 305
    print(f"验证集(已知类): {len(val_known)} 张图像")               # 45
    print(f"测试集(已知类): {len(test_known)} 张图像")              # 90
    print(f"未知类(来自 Food-101): {len(test_unknown)} 张图像")     # 250

    print("\n" + "=" * 60)
    print(f"2. 加载 CLIP 模型: {args.model_name}")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # if args.temperature and args.temperature > 0:
    #     with torch.no_grad():
    #         logit_scale = np.log(1.0 / args.temperature)
    #         logit_scale = np.clip(logit_scale, np.log(1 / 100), np.log(100))
    #         model.logit_scale.data = torch.tensor(
    #             [logit_scale], device=device, dtype=torch.float32
    #         )
    #     print(f"使用 temperature={args.temperature} 重设 logit_scale")
    # else:
    #     print("保持 CLIP 预训练模型默认 logit_scale")

    print("\n" + "=" * 60)
    print("3. 构建统一候选标签，并计算文本特征")
    all_class_names = KNOWN_CLASSES + UNKNOWN_CLASSES_FOOD101
    all_prompts = get_class_prompts(all_class_names, ensemble=args.prompt_ensemble)

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

    val_loader = DataLoader(
        val_known, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_known, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    unknown_loader = DataLoader(
        test_unknown, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print("\n" + "=" * 60)
    print("4. 已知类 closed-set zero-shot 诊断评估")
    val_known_closed = evaluate_zero_shot(
        model, 
        known_text_features, 
        val_loader, 
        device, 
        label_offset=0, 
        desc="验证集已知类"
    )
    test_known_closed = evaluate_zero_shot(
        model, 
        known_text_features, 
        test_loader, 
        device, 
        label_offset=0, 
        desc="测试集已知类"
    )

    print(f"验证集 known-only Top-1 准确率: {val_known_closed['accuracy'] * 100:.2f}%")
    print(f"测试集 known-only Top-1 准确率: {test_known_closed['accuracy'] * 100:.2f}%")

    closed_report = classification_report(
        test_known_closed["labels"],
        test_known_closed["preds"],
        labels=list(range(len(KNOWN_CLASSES))),
        target_names=KNOWN_CLASSES,
        zero_division=0,
    )
    print("\n===== 已知类 closed-set 分类报告 =====")
    print(closed_report)

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
    for key, value in open_set_results.items():
        if key in {"all_true_labels", "all_pred_labels"}:
            continue
        if isinstance(value, float):
            print(f"{key}: {value * 100:.2f}%")
        else:
            print(f"{key}: {value}")

    open_report = classification_report(
        open_set_results["all_true_labels"],
        open_set_results["all_pred_labels"],
        labels=list(range(len(all_class_names))),
        target_names=all_class_names,
        zero_division=0,
    )
    open_cm = confusion_matrix(
        open_set_results["all_true_labels"],
        open_set_results["all_pred_labels"],
        labels=list(range(len(all_class_names))),
    )

    print("\n===== 统一 10 类分类报告 =====")
    print(open_report)
    print("\n===== 统一 10 类混淆矩阵 =====")
    print(open_cm)

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

        file_obj.write("===== 方法说明 =====\n")
        file_obj.write(
            "本实验将 known classes 与 unknown classes 合并为统一候选标签集合，"
            "计算每张图像与所有候选文本标签的 CLIP 相似度，并直接选择相似度最高的类别。"
            "本版本不使用阈值拒识，也不进行阈值扫描。\n\n"
        )

        file_obj.write("===== 已知类 known-only closed-set 诊断评估 =====\n")
        file_obj.write(f"验证集准确率: {val_known_closed['accuracy'] * 100:.2f}%\n")
        file_obj.write(f"测试集准确率: {test_known_closed['accuracy'] * 100:.2f}%\n")
        file_obj.write(closed_report + "\n")

        file_obj.write("===== 统一 10 类 zero-shot 开放集评估 =====\n")
        _write_dict_results(file_obj, open_set_results)
        file_obj.write("\n===== 统一 10 类分类报告 =====\n")
        file_obj.write(open_report + "\n")
        file_obj.write("\n===== 统一 10 类混淆矩阵 =====\n")
        file_obj.write(str(open_cm) + "\n")

    print(f"结果已保存到: {result_path}")
    print("实验完成！")


if __name__ == "__main__":
    main()
