import argparse
import json
import random
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


CLASS_ORDER = ["Noodles", "Rice", "bottled water", "canned goods", "combo"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEED = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_model(num_classes: int) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _filename_order_key(sample_path: str) -> tuple[str, int, str]:
    path = Path(sample_path)
    match = re.search(r"(\d+)(?=\D*$)", path.stem)
    number = int(match.group(1)) if match else 0
    return path.parent.name.lower(), number, path.name.lower()


def build_split_indices(samples, targets, strategy: str, seed: int = SEED):
    if strategy == "lab8_random":
        indices = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed)).tolist()
        n_train = int(0.60 * len(indices))
        n_val = int(0.20 * len(indices))
        return indices[:n_train], indices[n_train : n_train + n_val], indices[n_train + n_val :]

    if strategy != "blocked_by_filename":
        raise ValueError(f"Unknown split strategy: {strategy}")

    train_indices, val_indices, test_indices = [], [], []
    for class_idx in sorted(set(targets)):
        class_indices = [idx for idx, target in enumerate(targets) if target == class_idx]
        class_indices = sorted(class_indices, key=lambda idx: _filename_order_key(samples[idx][0]))
        n_total = len(class_indices)
        n_train = int(0.60 * n_total)
        n_val = int(0.20 * n_total)
        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train : n_train + n_val])
        test_indices.extend(class_indices[n_train + n_val :])
    return train_indices, val_indices, test_indices


def build_loaders(data_dir: Path, img_size: int, batch_size: int, num_workers: int, split_strategy: str):
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop(img_size, scale=(0.72, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.15, hue=0.03)],
                p=0.85,
            ),
            transforms.RandomRotation(degrees=12),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.35),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.20),
            transforms.RandomAutocontrast(p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
            transforms.RandomErasing(p=0.18, scale=(0.02, 0.10), ratio=(0.3, 3.3), value="random"),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    base_dataset = ImageFolder(data_dir, transform=eval_transform)
    if base_dataset.classes != CLASS_ORDER:
        raise RuntimeError(f"Class order mismatch. Expected {CLASS_ORDER}, got {base_dataset.classes}")

    train_indices, val_indices, test_indices = build_split_indices(
        base_dataset.samples,
        base_dataset.targets,
        strategy=split_strategy,
        seed=SEED,
    )

    train_dataset = ImageFolder(data_dir, transform=train_transform)
    eval_dataset = ImageFolder(data_dir, transform=eval_transform)
    train_targets = [base_dataset.targets[i] for i in train_indices]
    class_counts = np.bincount(train_targets, minlength=len(CLASS_ORDER))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[base_dataset.targets[i]] for i in train_indices]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(eval_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(eval_dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return base_dataset, train_indices, val_indices, test_indices, train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, scaler, device, training: bool):
    model.train(training)
    total_loss = 0.0
    all_labels, all_preds = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.item()) * images.size(0)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    recalls = recall_score(all_labels, all_preds, average=None, labels=list(range(len(CLASS_ORDER))), zero_division=0)
    return avg_loss, accuracy, macro_f1, float(np.min(recalls))


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    labels, preds, confidences = [], [], []
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            probs = torch.softmax(logits.float(), dim=1)
        labels.extend(y.numpy().tolist())
        preds.extend(probs.argmax(dim=1).cpu().numpy().tolist())
        confidences.extend(probs.max(dim=1).values.cpu().numpy().tolist())
    return np.array(labels), np.array(preds), np.array(confidences)


def save_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_ORDER)), labels=CLASS_ORDER, rotation=35, ha="right")
    ax.set_yticks(range(len(CLASS_ORDER)), labels=CLASS_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_deployment_metadata(deployment_dir: Path, img_size: int, accuracy: float, macro_f1: float, sample_source: Path) -> None:
    (deployment_dir / "labels.txt").write_text("\n".join(CLASS_ORDER) + "\n", encoding="utf-8")
    (deployment_dir / "preprocessing.txt").write_text(
        f"Input size: {img_size}x{img_size} RGB\n"
        f"Input tensor for PyTorch training: NCHW float32, shape (1, 3, {img_size}, {img_size})\n"
        f"Input tensor for converted TFLite model: NHWC float32, shape (1, {img_size}, {img_size}, 3) unless converter reports otherwise\n"
        "Resize: center-crop square then resize with area/PIL-style interpolation\n"
        "Color order: RGB\n"
        "Scale: uint8 pixels to [0, 1]\n"
        "Normalization mean: [0.485, 0.456, 0.406]\n"
        "Normalization std: [0.229, 0.224, 0.225]\n",
        encoding="utf-8",
    )
    Image.open(sample_source).convert("RGB").save(deployment_dir / "sample_input.jpg")
    model_card = "\n".join(
        [
            "# Lab 10 MobileNetV3-small Model Card",
            "",
            "Source model: GPU retrained PyTorch MobileNetV3-small",
            "",
            "Dataset: Lab04 EDA Bias Dataset copied from Lab 8; no new Pi-camera training images were collected.",
            "",
            "Task type: 5-class image classification",
            "",
            "Classes: " + ", ".join(CLASS_ORDER),
            "",
            f"Input shape: RGB {img_size}x{img_size}, normalized with ImageNet mean/std",
            "",
            "Output format: logits over 5 classes in labels.txt order",
            "",
            "Selection rule: best validation macro F1 plus worst-class recall score",
            "",
            f"Test accuracy: {accuracy:.4f}",
            "",
            f"Macro F1: {macro_f1:.4f}",
            "",
            "Deployment note: convert the saved PyTorch checkpoint to TFLite before replacing model.tflite on the Raspberry Pi.",
        ]
    )
    (deployment_dir / "model_card.md").write_text(model_card + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Lab 10 classifier on CUDA and write evaluation artifacts.")
    parser.add_argument("--data-dir", type=Path, default=Path("ml-perception-labs/lab08_finetuning/data/raw"))
    parser.add_argument("--lab10-root", type=Path, default=Path("ml-perception-labs/lab10_rpi_deployment"))
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows to avoid DataLoader spawn overhead.")
    parser.add_argument(
        "--split-strategy",
        choices=["lab8_random", "blocked_by_filename"],
        default="blocked_by_filename",
        help="Use blocked_by_filename for stricter unseen-batch evaluation; lab8_random reproduces earlier Lab 8/10 numbers.",
    )
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU training. Not recommended for this lab.")
    args = parser.parse_args()

    seed_everything(SEED)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA GPU is required. Re-run on the GPU machine or pass --allow-cpu only for debugging.")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    deployment_dir = args.lab10_root / "deployment_package"
    figures_dir = args.lab10_root / "outputs" / "figures"
    tables_dir = args.lab10_root / "outputs" / "tables"
    logs_dir = args.lab10_root / "outputs" / "logs"
    for path in [deployment_dir, figures_dir, tables_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    base_dataset, _, _, test_indices, train_loader, val_loader, test_loader = build_loaders(
        args.data_dir, args.img_size, args.batch_size, args.num_workers, args.split_strategy
    )

    model = build_model(len(CLASS_ORDER)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")

    best_score = -1.0
    best_path = deployment_dir / "lab10_mobilenetv3_small_best.pth"
    history = []
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc, train_f1, train_worst_recall = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, training=True
        )
        val_loss, val_acc, val_f1, val_worst_recall = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, training=False
        )
        scheduler.step()
        score = 0.70 * val_f1 + 0.30 * val_worst_recall
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "train_worst_recall": train_worst_recall,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "val_worst_recall": val_worst_recall,
            "selection_score": score,
            "seconds": time.perf_counter() - start,
        }
        history.append(row)
        print(json.dumps(row, indent=2))
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(tables_dir / "lab10_training_history.csv", index=False)

    model.load_state_dict(torch.load(best_path, map_location=device))
    y_true, y_pred, y_conf = collect_predictions(model, test_loader, device)
    report = classification_report(y_true, y_pred, target_names=CLASS_ORDER, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(tables_dir / "lab10_test_classification_report.csv")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_ORDER))))
    pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(tables_dir / "lab10_test_confusion_matrix.csv")
    save_confusion_matrix(cm, figures_dir / "lab10_test_confusion_matrix.png")

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    worst_recall = min(recall_score(y_true, y_pred, average=None, labels=list(range(len(CLASS_ORDER))), zero_division=0))
    summary = {
        "model": "MobileNetV3-small",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "img_size": args.img_size,
        "split_strategy": args.split_strategy,
        "test_accuracy": accuracy,
        "macro_f1": macro_f1,
        "worst_class_recall": worst_recall,
        "checkpoint": str(best_path.resolve()),
    }
    (logs_dir / "lab10_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_deployment_metadata(deployment_dir, args.img_size, accuracy, macro_f1, Path(base_dataset.samples[test_indices[0]][0]))

    print("\nFinal test metrics")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved checkpoint: {best_path}")
    print(f"Saved report table: {tables_dir / 'lab10_test_classification_report.csv'}")
    print("Next step: run the notebook conversion cell or Lab 9 conversion flow to produce a new model.tflite.")


if __name__ == "__main__":
    main()
