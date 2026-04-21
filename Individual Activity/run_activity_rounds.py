#!/usr/bin/env python3
"""All-in-one Hyperparameter Tuning Activity runner.

What this script does automatically:
1) Installs missing Python dependencies.
2) Detects available GPU and uses it when supported by PyTorch.
3) Uses torchvision Fashion-MNIST by default (optional Kaggle CSV mode).
4) Runs Round 1 to Round 7 experiments.
5) Preserves all results in JSON + Markdown + CSV.
6) Saves visual proof images for each run (curves + terminal-style logs + summaries).
7) Saves round-level result table images and final confusion matrix image.

Usage:
    python "Individual Activity/run_activity_rounds.py"
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("PIL", "Pillow"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("tqdm", "tqdm"),
]


def _install_with_pip(packages: list[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    print(f"[setup] Installing missing packages: {', '.join(packages)}")
    subprocess.check_call(cmd)


def ensure_runtime_dependencies() -> None:
    missing: list[str] = []
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        try:
            _install_with_pip(missing)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                "Failed to auto-install required dependencies. "
                "Please run: pip install " + " ".join(missing)
            ) from exc


if __name__ == "__main__":
    if "--help" not in sys.argv and "-h" not in sys.argv:
        print("=====================================================")
        print(" Hyperparameter Tuning Activity Setup Menu")
        print("=====================================================")
        print("1) Auto run and forget (install dependencies and run)")
        print("2) Manual run (run without installing dependencies)")
        print("3) Install all dependencies ONLY")
        print("=====================================================")
        while True:
            try:
                choice = input("Enter choice (1, 2, or 3): ").strip()
            except KeyboardInterrupt:
                print("\nExiting.")
                sys.exit(0)
            
            if choice in {'1', '2', '3'}:
                break
            print("Invalid choice. Please enter 1, 2, or 3.")
        
        if choice == '1':
            ensure_runtime_dependencies()
        elif choice == '2':
            pass
        elif choice == '3':
            ensure_runtime_dependencies()
            print("Dependencies installed successfully. Exiting as requested.")
            sys.exit(0)
    else:
        ensure_runtime_dependencies()
else:
    ensure_runtime_dependencies()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from tqdm import tqdm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

TERMINAL_BG = (12, 12, 12)
TERMINAL_FG = (233, 236, 239)
TERMINAL_ACCENT = (80, 170, 255)
TERMINAL_PADDING = 22


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    freeze_backbone: bool
    dropout: float
    learning_rate: float
    batch_size: int
    epochs: int
    optimizer: str
    scheduler: str | None
    train_size: int
    use_augmentation: bool


@dataclass(frozen=True)
class DatasetBundle:
    source: str
    train_plain: Dataset
    train_augmented: Dataset
    test_set: Dataset
    notes: list[str]


class KaggleFashionMNISTDataset(Dataset):
    """Fashion-MNIST dataset reader for Kaggle CSV files."""

    def __init__(self, csv_path: Path, transform: transforms.Compose | None = None):
        df = pd.read_csv(csv_path)
        if df.shape[1] < 785:
            raise ValueError(f"Unexpected CSV shape for Fashion-MNIST: {df.shape}")

        self.labels = df.iloc[:, 0].to_numpy(dtype=np.int64)
        pixels = df.iloc[:, 1:].to_numpy(dtype=np.uint8)
        if pixels.shape[1] != 784:
            raise ValueError("Expected 784 pixel columns in Kaggle CSV")

        self.images = pixels.reshape(-1, 28, 28)
        self.transform = transform

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.fromarray(self.images[index], mode="L")
        label = int(self.labels[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def basic_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(96),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def augmented_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(96),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def detect_nvidia_smi() -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(query, check=True, capture_output=True, text=True)
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        gpus = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append(
                    {
                        "name": parts[0],
                        "memory": parts[1],
                        "driver": parts[2],
                    }
                )
        return {"available": True, "gpus": gpus}
    except Exception:
        return {"available": False, "gpus": []}


def select_best_device() -> tuple[torch.device, dict[str, Any]]:
    info: dict[str, Any] = {
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "nvidia_smi": detect_nvidia_smi(),
    }

    # Rule: prefer dedicated NVIDIA GPU (CUDA). If unavailable, use CPU.
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        best_idx = 0
        best_mem = 0
        gpu_list: list[dict[str, Any]] = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            mem = int(props.total_memory)
            gpu_list.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(mem / (1024**3), 2),
                }
            )
            if mem > best_mem:
                best_mem = mem
                best_idx = idx

        torch.cuda.set_device(best_idx)
        selected = torch.device(f"cuda:{best_idx}")
        info["torch_cuda_devices"] = gpu_list
        info["selected"] = gpu_list[best_idx]
        return selected, info

    warning = None
    if info["nvidia_smi"]["available"] and not info["torch_cuda_available"]:
        warning = (
            "NVIDIA GPU detected by nvidia-smi but PyTorch CUDA is unavailable. "
            "Install CUDA-enabled torch build to use dedicated GPU."
        )
    info["warning"] = warning
    info["selected"] = {"name": "cpu"}
    return torch.device("cpu"), info


def find_kaggle_csvs(kaggle_dir: Path) -> tuple[Path | None, Path | None]:
    if not kaggle_dir.exists():
        return None, None

    csv_files = list(kaggle_dir.rglob("*.csv"))
    train_csv = None
    test_csv = None

    for csv_path in csv_files:
        lower = csv_path.name.lower()
        if "train" in lower and train_csv is None:
            train_csv = csv_path
        if "test" in lower and test_csv is None:
            test_csv = csv_path

    return train_csv, test_csv


def _kaggle_credentials_available() -> bool:
    """Return True if Kaggle API credentials look configured.

    The Kaggle SDK calls sys.exit() when credentials are missing, which raises
    SystemExit (not Exception) and would otherwise kill the whole script. We
    pre-check so we can cleanly fall back to torchvision without the noisy
    "You must authenticate..." message terminating the run.
    """
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True

    kaggle_json_candidates = [
        Path(os.environ.get("KAGGLE_CONFIG_DIR", "")) / "kaggle.json"
        if os.environ.get("KAGGLE_CONFIG_DIR")
        else None,
        Path.home() / ".kaggle" / "kaggle.json",
    ]
    for candidate in kaggle_json_candidates:
        if candidate is not None and candidate.exists():
            return True
    return False


def download_from_kaggle(data_dir: Path, notes: list[str]) -> Path | None:
    kaggle_dir = data_dir / "kaggle_fashionmnist"
    ensure_dir(kaggle_dir)

    train_csv, test_csv = find_kaggle_csvs(kaggle_dir)
    if train_csv is not None and test_csv is not None:
        notes.append(f"Kaggle dataset already present at: {kaggle_dir}")
        return kaggle_dir

    if not _kaggle_credentials_available():
        notes.append(
            "Kaggle credentials not found (no kaggle.json and no KAGGLE_USERNAME/KAGGLE_KEY "
            "environment variables). Falling back to torchvision dataset."
        )
        return None

    dataset_id = "zalando-research/fashionmnist"
    try:
        # Import lazily so default torchvision-only runs never touch Kaggle.
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        notes.append(f"Downloading Kaggle dataset: {dataset_id}")
        api.dataset_download_files(dataset_id, path=str(kaggle_dir), unzip=True, quiet=False)

        train_csv, test_csv = find_kaggle_csvs(kaggle_dir)
        if train_csv is None or test_csv is None:
            notes.append("Kaggle download completed but expected train/test CSV files were not found.")
            return None

        notes.append(f"Kaggle dataset downloaded to: {kaggle_dir}")
        return kaggle_dir
    except BaseException as exc:
        # Catch BaseException (not just Exception) because the Kaggle SDK may
        # call sys.exit() on auth failures, which raises SystemExit.
        notes.append(f"Kaggle download failed ({type(exc).__name__}: {exc}). Falling back to torchvision dataset.")
        return None


def load_datasets(data_dir: Path, prefer_kaggle: bool = False) -> DatasetBundle:
    notes: list[str] = []

    if prefer_kaggle:
        kaggle_dir = download_from_kaggle(data_dir, notes)
        if kaggle_dir is not None:
            train_csv, test_csv = find_kaggle_csvs(kaggle_dir)
            if train_csv is not None and test_csv is not None:
                train_plain = KaggleFashionMNISTDataset(train_csv, transform=basic_transform())
                train_aug = KaggleFashionMNISTDataset(train_csv, transform=augmented_transform())
                test_set = KaggleFashionMNISTDataset(test_csv, transform=basic_transform())
                notes.append("Using Kaggle CSV dataset for training/testing.")
                return DatasetBundle(
                    source="kaggle",
                    train_plain=train_plain,
                    train_augmented=train_aug,
                    test_set=test_set,
                    notes=notes,
                )
            notes.append("Kaggle files missing/invalid. Switching to torchvision source.")

    notes.append("Using torchvision.datasets.FashionMNIST (download=True).")
    train_plain_tv = datasets.FashionMNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=basic_transform(),
    )
    train_aug_tv = datasets.FashionMNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=augmented_transform(),
    )
    test_tv = datasets.FashionMNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=basic_transform(),
    )

    return DatasetBundle(
        source="torchvision",
        train_plain=train_plain_tv,
        train_augmented=train_aug_tv,
        test_set=test_tv,
        notes=notes,
    )


def build_model(device: torch.device, freeze_backbone: bool = True, dropout: float = 0.2) -> nn.Module:
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 10),
    )
    return model.to(device)


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def make_loaders(
    bundle: DatasetBundle,
    seed: int,
    train_size: int,
    batch_size: int,
    augment: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = bundle.train_augmented if augment else bundle.train_plain

    n_train = min(train_size, len(train_dataset))
    rng = np.random.default_rng(seed)
    all_idx = rng.permutation(len(train_dataset))

    n_val = int(0.1 * n_train)
    train_idx = all_idx[: n_train - n_val]
    val_idx = all_idx[n_train - n_val : n_train]

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(bundle.train_plain, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(bundle.test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


def build_optimizer(name: str, params: list[torch.nn.Parameter], learning_rate: float) -> optim.Optimizer:
    lower = name.lower()
    if lower == "adam":
        return optim.Adam(params, lr=learning_rate)
    if lower == "sgd":
        return optim.SGD(params, lr=learning_rate, momentum=0.9)
    if lower == "adamw":
        return optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(
    scheduler_type: str | None,
    optimizer: optim.Optimizer,
    epochs: int,
) -> optim.lr_scheduler._LRScheduler | None:
    if scheduler_type is None:
        return None
    if scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if scheduler_type == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    raise ValueError(f"Unknown scheduler: {scheduler_type}")


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    optimizer_name: str,
    scheduler_type: str | None,
    overall_pbar: tqdm | None = None,
) -> tuple[dict[str, list[float]], float, float, list[str]]:
    criterion = nn.CrossEntropyLoss()
    params_to_train = [p for p in model.parameters() if p.requires_grad]

    optimizer = build_optimizer(optimizer_name, params_to_train, learning_rate)
    scheduler = build_scheduler(scheduler_type, optimizer, epochs)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    logs: list[str] = []

    def emit(message: str) -> None:
        if overall_pbar is not None:
            # Keep the global bar stable while still showing live run output.
            tqdm.write(message)
        else:
            print(message)
        logs.append(message)

    emit(f"{'Epoch':>6} {'TrainLoss':>10} {'TrainAcc':>9} {'ValLoss':>9} {'ValAcc':>8} {'Time':>8}")
    emit("-" * 56)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()
        
        if overall_pbar is not None:
            overall_pbar.update(1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        emit(
            f"{epoch:>6} {train_loss:>10.4f} {train_acc * 100:>8.2f}% "
            f"{val_loss:>9.4f} {val_acc * 100:>7.2f}% {time.time() - t_epoch:>7.1f}s"
        )

    total_time = time.time() - t0
    _, test_acc = evaluate(model, test_loader, criterion, device)

    emit("-" * 56)
    emit(f"Training time : {total_time:.1f} seconds")
    emit(f"TEST accuracy : {test_acc * 100:.2f}%")

    final_gap = history["train_acc"][-1] - history["val_acc"][-1]
    if final_gap > 0.10:
        emit(f"OVERFITTING warning: train-val gap = {final_gap * 100:.2f} pp")
    elif history["train_acc"][-1] < 0.50:
        emit("UNDERFITTING warning: final train acc is below 50%")

    return history, test_acc, total_time, logs


def top_confusions(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    top_k: int = 10,
) -> list[dict[str, int | str]]:
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for true_label, pred_label in zip(labels, preds):
        cm[int(true_label), int(pred_label)] += 1

    mistakes: list[tuple[int, int, int]] = []
    for true_idx in range(n_classes):
        for pred_idx in range(n_classes):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count > 0:
                mistakes.append((count, true_idx, pred_idx))

    mistakes.sort(reverse=True, key=lambda item: item[0])

    output: list[dict[str, int | str]] = []
    for count, true_idx, pred_idx in mistakes[:top_k]:
        output.append(
            {
                "count": count,
                "true_label": true_idx,
                "true_class": class_names[true_idx],
                "pred_label": pred_idx,
                "pred_class": class_names[pred_idx],
            }
        )

    return output


def confusion_matrix_array(labels: np.ndarray, preds: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(labels, preds):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def _load_monospace_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/Consolas.ttf",
        "DejaVuSansMono.ttf",
    ]
    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def save_text_proof_image(
    text: str,
    output_path: Path,
    title: str | None = None,
    font_size: int = 18,
    max_chars: int = 120,
) -> None:
    ensure_dir(output_path.parent)

    wrapped_lines: list[str] = []
    if title:
        wrapped_lines.extend(textwrap.wrap(title, width=max_chars) or [title])
        wrapped_lines.append("")

    for line in text.splitlines():
        parts = textwrap.wrap(line, width=max_chars, replace_whitespace=False, drop_whitespace=False)
        if parts:
            wrapped_lines.extend(parts)
        else:
            wrapped_lines.append("")

    if not wrapped_lines:
        wrapped_lines = ["[empty]"]

    font = _load_monospace_font(font_size)
    sample_bbox = font.getbbox("Ag")
    line_h = max(22, (sample_bbox[3] - sample_bbox[1]) + 6)

    width = TERMINAL_PADDING * 2 + 16 * max_chars
    height = TERMINAL_PADDING * 2 + line_h * len(wrapped_lines)

    image = Image.new("RGB", (width, height), TERMINAL_BG)
    draw = ImageDraw.Draw(image)

    y = TERMINAL_PADDING
    for idx, line in enumerate(wrapped_lines):
        color = TERMINAL_ACCENT if title and idx < len(textwrap.wrap(title, width=max_chars) or [title]) else TERMINAL_FG
        draw.text((TERMINAL_PADDING, y), line, fill=color, font=font)
        y += line_h

    image.save(output_path)


def save_history_plot(history: dict[str, list[float]], title: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)

    epochs = np.arange(1, len(history["train_acc"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(epochs, np.array(history["train_acc"]) * 100, marker="o", label="Train")
    ax1.plot(epochs, np.array(history["val_acc"]) * 100, marker="s", label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["train_loss"], marker="o", label="Train")
    ax2.plot(epochs, history["val_loss"], marker="s", label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_round_table_image(round_name: str, round_results: list[dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)

    rows: list[list[str]] = []
    for item in round_results:
        rows.append(
            [
                str(item["run_id"]),
                f"{item['test_acc'] * 100:.2f}%",
                f"{item['final_train_acc'] * 100:.2f}%",
                f"{item['final_val_acc'] * 100:.2f}%",
                f"{item['train_val_gap'] * 100:.2f} pp",
                f"{item['training_time_seconds']:.1f}s",
            ]
        )

    headers = ["Run", "Test", "Train", "Val", "Gap", "Time"]

    fig, ax = plt.subplots(figsize=(10, max(2.8, 1.0 + 0.55 * len(rows))))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)

    ax.set_title(f"{round_name.upper()} Results", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_plot(matrix: np.ndarray, class_names: list[str], output_path: Path) -> None:
    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Round 7 Confusion Matrix")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_rounds() -> dict[str, list[RunConfig]]:
    return {
        "round1": [
            RunConfig("1", True, 0.2, 0.00001, 64, 2, "adam", None, 5000, False),
        ],
        "round2": [
            RunConfig("2a", True, 0.2, 1.0, 64, 3, "adam", None, 5000, False),
            RunConfig("2b", True, 0.2, 0.1, 64, 3, "adam", None, 5000, False),
            RunConfig("2c", True, 0.2, 0.01, 64, 3, "adam", None, 5000, False),
            RunConfig("2d", True, 0.2, 0.001, 64, 3, "adam", None, 5000, False),
            RunConfig("2e", True, 0.2, 0.0001, 64, 3, "adam", None, 5000, False),
        ],
        "round3": [
            RunConfig("3a", True, 0.2, 0.001, 64, 3, "adam", None, 5000, False),
            RunConfig("3b", True, 0.2, 0.001, 64, 8, "adam", None, 5000, False),
            RunConfig("3c", True, 0.2, 0.001, 64, 5, "adam", None, 10000, False),
            RunConfig("3d", True, 0.2, 0.001, 64, 5, "adam", None, 20000, False),
        ],
        "round4": [
            RunConfig("4a", True, 0.2, 0.001, 64, 5, "adam", None, 10000, False),
            RunConfig("4b", False, 0.2, 0.001, 64, 5, "adam", None, 10000, False),
            RunConfig("4c", False, 0.2, 0.0001, 64, 5, "adam", None, 10000, False),
            RunConfig("4d", False, 0.2, 0.00001, 64, 5, "adam", None, 10000, False),
        ],
        "round5": [
            RunConfig("5a", False, 0.2, 0.0001, 64, 8, "adam", None, 10000, False),
            RunConfig("5b", False, 0.2, 0.0001, 64, 8, "adam", None, 10000, True),
            RunConfig("5c", False, 0.2, 0.0001, 64, 12, "adam", None, 10000, True),
        ],
        "round6": [
            RunConfig("6a", False, 0.2, 0.0001, 64, 10, "adam", None, 10000, True),
            RunConfig("6b", False, 0.2, 0.0001, 64, 10, "adam", "step", 10000, True),
            RunConfig("6c", False, 0.2, 0.0001, 64, 10, "adam", "cosine", 10000, True),
        ],
        "round7": [
            RunConfig("7", False, 0.2, 0.0001, 64, 10, "adam", "step", 10000, True),
        ],
    }


def assert_strict_worksheet_adherence(
    rounds: dict[str, list[RunConfig]],
    worksheet_path: Path,
) -> dict[str, Any]:
    if not worksheet_path.exists():
        raise FileNotFoundError(f"Worksheet file not found: {worksheet_path}")

    worksheet_text = worksheet_path.read_text(encoding="utf-8")
    required_headers = [
        "## Round 1 - Baseline",
        "## Round 2 - Learning Rate",
        "## Round 3 - More Data and More Epochs",
        "## Round 4 - Fine-Tuning (unfreeze the backbone)",
        "## Round 5 - Data Augmentation",
        "## Round 6 - Learning Rate Scheduler",
        "## Round 7 - Your Best Configuration",
    ]
    missing_headers = [header for header in required_headers if header not in worksheet_text]
    if missing_headers:
        raise ValueError(f"Worksheet is missing required sections: {missing_headers}")

    expected_ids = {
        "round1": ["1"],
        "round2": ["2a", "2b", "2c", "2d", "2e"],
        "round3": ["3a", "3b", "3c", "3d"],
        "round4": ["4a", "4b", "4c", "4d"],
        "round5": ["5a", "5b", "5c"],
        "round6": ["6a", "6b", "6c"],
        "round7": ["7"],
    }

    for round_name, run_ids in expected_ids.items():
        if round_name not in rounds:
            raise ValueError(f"Missing round in script config: {round_name}")
        actual_ids = [cfg.run_id for cfg in rounds[round_name]]
        if actual_ids != run_ids:
            raise ValueError(
                f"Run IDs mismatch for {round_name}. Expected {run_ids}, got {actual_ids}."
            )

    # Strict config check for rounds where worksheet defines explicit values.
    expected_configs: dict[str, dict[str, Any]] = {
        "1": {
            "freeze_backbone": True,
            "dropout": 0.2,
            "learning_rate": 0.00001,
            "batch_size": 64,
            "epochs": 2,
            "optimizer": "adam",
            "scheduler": None,
            "train_size": 5000,
            "use_augmentation": False,
        },
        "2a": {"freeze_backbone": True, "dropout": 0.2, "learning_rate": 1.0, "batch_size": 64, "epochs": 3, "optimizer": "adam", "scheduler": None, "train_size": 5000, "use_augmentation": False},
        "2b": {"freeze_backbone": True, "dropout": 0.2, "learning_rate": 0.1, "batch_size": 64, "epochs": 3, "optimizer": "adam", "scheduler": None, "train_size": 5000, "use_augmentation": False},
        "2c": {"freeze_backbone": True, "dropout": 0.2, "learning_rate": 0.01, "batch_size": 64, "epochs": 3, "optimizer": "adam", "scheduler": None, "train_size": 5000, "use_augmentation": False},
        "2d": {"freeze_backbone": True, "dropout": 0.2, "learning_rate": 0.001, "batch_size": 64, "epochs": 3, "optimizer": "adam", "scheduler": None, "train_size": 5000, "use_augmentation": False},
        "2e": {"freeze_backbone": True, "dropout": 0.2, "learning_rate": 0.0001, "batch_size": 64, "epochs": 3, "optimizer": "adam", "scheduler": None, "train_size": 5000, "use_augmentation": False},
        "3a": {"freeze_backbone": True, "dropout": 0.2, "learning_rate": 0.001, "batch_size": 64, "epochs": 3, "optimizer": "adam", "scheduler": None, "train_size": 5000, "use_augmentation": False},
        "3b": {"freeze_backbone": True, "dropout": 0.2, "learning_rate": 0.001, "batch_size": 64, "epochs": 8, "optimizer": "adam", "scheduler": None, "train_size": 5000, "use_augmentation": False},
        "3c": {"freeze_backbone": True, "dropout": 0.2, "learning_rate": 0.001, "batch_size": 64, "epochs": 5, "optimizer": "adam", "scheduler": None, "train_size": 10000, "use_augmentation": False},
        "3d": {"freeze_backbone": True, "dropout": 0.2, "learning_rate": 0.001, "batch_size": 64, "epochs": 5, "optimizer": "adam", "scheduler": None, "train_size": 20000, "use_augmentation": False},
        "4a": {"freeze_backbone": True, "dropout": 0.2, "learning_rate": 0.001, "batch_size": 64, "epochs": 5, "optimizer": "adam", "scheduler": None, "train_size": 10000, "use_augmentation": False},
        "4b": {"freeze_backbone": False, "dropout": 0.2, "learning_rate": 0.001, "batch_size": 64, "epochs": 5, "optimizer": "adam", "scheduler": None, "train_size": 10000, "use_augmentation": False},
        "4c": {"freeze_backbone": False, "dropout": 0.2, "learning_rate": 0.0001, "batch_size": 64, "epochs": 5, "optimizer": "adam", "scheduler": None, "train_size": 10000, "use_augmentation": False},
        "4d": {"freeze_backbone": False, "dropout": 0.2, "learning_rate": 0.00001, "batch_size": 64, "epochs": 5, "optimizer": "adam", "scheduler": None, "train_size": 10000, "use_augmentation": False},
        "5a": {"freeze_backbone": False, "dropout": 0.2, "learning_rate": 0.0001, "batch_size": 64, "epochs": 8, "optimizer": "adam", "scheduler": None, "train_size": 10000, "use_augmentation": False},
        "5b": {"freeze_backbone": False, "dropout": 0.2, "learning_rate": 0.0001, "batch_size": 64, "epochs": 8, "optimizer": "adam", "scheduler": None, "train_size": 10000, "use_augmentation": True},
        "5c": {"freeze_backbone": False, "dropout": 0.2, "learning_rate": 0.0001, "batch_size": 64, "epochs": 12, "optimizer": "adam", "scheduler": None, "train_size": 10000, "use_augmentation": True},
        "6a": {"freeze_backbone": False, "dropout": 0.2, "learning_rate": 0.0001, "batch_size": 64, "epochs": 10, "optimizer": "adam", "scheduler": None, "train_size": 10000, "use_augmentation": True},
        "6b": {"freeze_backbone": False, "dropout": 0.2, "learning_rate": 0.0001, "batch_size": 64, "epochs": 10, "optimizer": "adam", "scheduler": "step", "train_size": 10000, "use_augmentation": True},
        "6c": {"freeze_backbone": False, "dropout": 0.2, "learning_rate": 0.0001, "batch_size": 64, "epochs": 10, "optimizer": "adam", "scheduler": "cosine", "train_size": 10000, "use_augmentation": True},
    }

    actual_by_id: dict[str, RunConfig] = {
        cfg.run_id: cfg for round_cfgs in rounds.values() for cfg in round_cfgs
    }
    for run_id, expected in expected_configs.items():
        actual_cfg = actual_by_id[run_id]
        actual = {
            "freeze_backbone": actual_cfg.freeze_backbone,
            "dropout": actual_cfg.dropout,
            "learning_rate": actual_cfg.learning_rate,
            "batch_size": actual_cfg.batch_size,
            "epochs": actual_cfg.epochs,
            "optimizer": actual_cfg.optimizer,
            "scheduler": actual_cfg.scheduler,
            "train_size": actual_cfg.train_size,
            "use_augmentation": actual_cfg.use_augmentation,
        }
        if actual != expected:
            raise ValueError(
                f"Config mismatch for run {run_id}. Expected {expected}, got {actual}."
            )

    # Round 7 is intentionally user-selected in the worksheet, so only ensure it's valid and complete.
    round7_cfg = actual_by_id["7"]
    if round7_cfg.optimizer not in {"adam", "sgd", "adamw"}:
        raise ValueError("Round 7 optimizer must be one of: adam, sgd, adamw")
    if round7_cfg.scheduler not in {None, "step", "cosine"}:
        raise ValueError("Round 7 scheduler must be one of: None, step, cosine")

    return {
        "worksheet_path": str(worksheet_path.resolve()),
        "required_sections_ok": True,
        "round_ids_ok": True,
        "round1_to_round6_config_ok": True,
        "round7_valid": True,
    }


def write_picture_index(output_path: Path, items: list[dict[str, str]]) -> None:
    lines = [
        "# Picture Index",
        "",
        "| Type | Round | Run | Path |",
        "| --- | --- | --- | --- |",
    ]

    for item in items:
        lines.append(
            f"| {item['type']} | {item.get('round', '')} | {item.get('run_id', '')} | {item['path']} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def verify_picture_structure(items: list[dict[str, str]]) -> None:
    for item in items:
        image_path = Path(item["path"])
        if image_path.suffix.lower() != ".png":
            raise ValueError(f"Non-PNG image artifact found: {image_path}")
        if not image_path.exists():
            raise FileNotFoundError(f"Missing picture artifact: {image_path}")


def automatic_note(result: dict[str, Any]) -> str:
    gap = float(result["train_val_gap"])
    train_acc = float(result["final_train_acc"])
    test_acc = float(result["test_acc"])

    if train_acc < 0.50:
        state = "underfitting"
    elif gap > 0.10:
        state = "overfitting risk"
    else:
        state = "balanced fit"

    return (
        f"{state}; final test accuracy={test_acc * 100:.2f}% "
        f"(train-val gap {gap * 100:.2f} pp)."
    )


def run_one_config(
    cfg: RunConfig,
    bundle: DatasetBundle,
    device: torch.device,
    seed: int,
    overall_pbar: tqdm | None = None,
) -> tuple[dict[str, Any], nn.Module, DataLoader, list[str]]:
    set_seed(seed)

    preface = [
        "=" * 88,
        f"Running {cfg.run_id}",
        f"Config: {asdict(cfg)}",
    ]
    if overall_pbar is not None:
        overall_pbar.set_postfix_str(f"run={cfg.run_id}")
    for line in preface:
        if overall_pbar is None:
            print(line)

    train_loader, val_loader, test_loader = make_loaders(
        bundle=bundle,
        seed=seed,
        train_size=cfg.train_size,
        batch_size=cfg.batch_size,
        augment=cfg.use_augmentation,
    )

    model = build_model(device=device, freeze_backbone=cfg.freeze_backbone, dropout=cfg.dropout)
    trainable_params, total_params = count_trainable_params(model)

    history, test_acc, training_time, train_logs = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        optimizer_name=cfg.optimizer,
        scheduler_type=cfg.scheduler,
        overall_pbar=overall_pbar,
    )

    final_train_acc = history["train_acc"][-1]
    final_val_acc = history["val_acc"][-1]
    gap = final_train_acc - final_val_acc

    result: dict[str, Any] = {
        "run_id": cfg.run_id,
        "config": asdict(cfg),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": trainable_params / total_params,
        "history": history,
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
        "train_val_gap": gap,
        "test_acc": test_acc,
        "training_time_seconds": training_time,
        "auto_note": automatic_note(
            {
                "train_val_gap": gap,
                "final_train_acc": final_train_acc,
                "test_acc": test_acc,
            }
        ),
    }

    full_log = preface + train_logs
    return result, model, test_loader, full_log


def save_run_artifacts(
    run_result: dict[str, Any],
    run_logs: list[str],
    run_dir: Path,
) -> dict[str, str]:
    ensure_dir(run_dir)

    curves_path = run_dir / "curves.png"
    terminal_png_path = run_dir / "terminal_proof.png"
    summary_png_path = run_dir / "summary.png"
    terminal_txt_path = run_dir / "terminal_log.txt"

    save_history_plot(
        history=run_result["history"],
        title=f"Run {run_result['run_id']} | Test {run_result['test_acc'] * 100:.2f}%",
        output_path=curves_path,
    )

    terminal_txt_path.write_text("\n".join(run_logs), encoding="utf-8")

    save_text_proof_image(
        text="\n".join(run_logs),
        output_path=terminal_png_path,
        title=f"Terminal Proof - Run {run_result['run_id']}",
        font_size=16,
        max_chars=130,
    )

    summary_lines = [
        f"Run ID: {run_result['run_id']}",
        f"Test accuracy: {run_result['test_acc'] * 100:.2f}%",
        f"Final train accuracy: {run_result['final_train_acc'] * 100:.2f}%",
        f"Final validation accuracy: {run_result['final_val_acc'] * 100:.2f}%",
        f"Train-val gap: {run_result['train_val_gap'] * 100:.2f} pp",
        f"Training time: {run_result['training_time_seconds']:.1f}s",
        f"Trainable params: {run_result['trainable_params']:,}/{run_result['total_params']:,}",
        f"Auto note: {run_result['auto_note']}",
    ]
    save_text_proof_image(
        text="\n".join(summary_lines),
        output_path=summary_png_path,
        title=f"Run {run_result['run_id']} Summary",
        font_size=18,
        max_chars=105,
    )

    return {
        "run_dir": str(run_dir.resolve()),
        "curves_png": str(curves_path.resolve()),
        "terminal_png": str(terminal_png_path.resolve()),
        "summary_png": str(summary_png_path.resolve()),
        "terminal_txt": str(terminal_txt_path.resolve()),
    }


def flatten_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in results:
        cfg = item["config"]
        rows.append(
            {
                "run_id": item["run_id"],
                "test_acc": item["test_acc"],
                "final_train_acc": item["final_train_acc"],
                "final_val_acc": item["final_val_acc"],
                "train_val_gap": item["train_val_gap"],
                "training_time_seconds": item["training_time_seconds"],
                "freeze_backbone": cfg["freeze_backbone"],
                "dropout": cfg["dropout"],
                "learning_rate": cfg["learning_rate"],
                "batch_size": cfg["batch_size"],
                "epochs": cfg["epochs"],
                "optimizer": cfg["optimizer"],
                "scheduler": cfg["scheduler"],
                "train_size": cfg["train_size"],
                "use_augmentation": cfg["use_augmentation"],
                "auto_note": item["auto_note"],
            }
        )
    return pd.DataFrame(rows)


def write_summary_markdown(
    output_path: Path,
    payload: dict[str, Any],
    round_order: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# Activity Results Summary")
    lines.append("")
    lines.append(f"Generated UTC: {payload['meta']['created_utc']}")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Device selected: {payload['meta']['device']}")
    lines.append(f"- Torch: {payload['meta']['torch']}")
    lines.append(f"- Torchvision: {payload['meta']['torchvision']}")
    lines.append(f"- Dataset source used: {payload['meta']['dataset_source']}")
    for note in payload["meta"]["dataset_notes"]:
        lines.append(f"- Dataset note: {note}")
    warning = payload["meta"]["device_info"].get("warning")
    if warning:
        lines.append(f"- Device warning: {warning}")

    by_id = {item["run_id"]: item for item in payload["results"]}

    for round_name in round_order:
        lines.append("")
        lines.append(f"## {round_name.upper()}")
        lines.append("")
        lines.append("| Run | Test | Train | Val | Gap | Time | Note |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for run_id in payload["rounds"][round_name]:
            row = by_id[run_id]
            lines.append(
                "| "
                f"{run_id} | "
                f"{row['test_acc'] * 100:.2f}% | "
                f"{row['final_train_acc'] * 100:.2f}% | "
                f"{row['final_val_acc'] * 100:.2f}% | "
                f"{row['train_val_gap'] * 100:.2f} pp | "
                f"{row['training_time_seconds']:.1f}s | "
                f"{row['auto_note']} |"
            )

    lines.append("")
    lines.append("## Final")
    lines.append("")
    lines.append(f"- Round 1 test accuracy: {payload['round1_test_acc'] * 100:.2f}%")
    lines.append(f"- Round 7 test accuracy: {payload['round7']['test_acc'] * 100:.2f}%")
    lines.append(f"- Improvement over Round 1: {payload['improvement_over_round1_pp']:.2f} percentage points")

    if payload["round7"]["top_confusions"]:
        lines.append("")
        lines.append("### Top Round 7 Confusions")
        lines.append("")
        lines.append("| Count | True | Predicted |")
        lines.append("| ---: | --- | --- |")
        for row in payload["round7"]["top_confusions"]:
            lines.append(f"| {row['count']} | {row['true_class']} | {row['pred_class']} |")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_terminal_logs_markdown(output_path: Path, run_logs: dict[str, list[str]]) -> None:
    lines: list[str] = ["# Terminal Logs by Run", ""]

    for run_id in sorted(run_logs.keys(), key=lambda value: (len(value), value)):
        lines.append(f"## Run {run_id}")
        lines.append("")
        lines.append("```text")
        lines.extend(run_logs[run_id])
        lines.append("```")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Run all worksheet rounds with auto setup, dataset bootstrap, and visual proof export."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=workspace_root / "data",
        help="Dataset root folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "activity_results.json",
        help="Main JSON output path.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=script_dir / "activity_artifacts",
        help="Folder where plots/proof images/logs are saved.",
    )
    parser.add_argument(
        "--prefer-kaggle",
        action="store_true",
        help="Prefer Kaggle CSV download first (requires Kaggle auth); otherwise use torchvision by default.",
    )
    parser.add_argument(
        "--skip-kaggle",
        action="store_true",
        help="Skip Kaggle dataset attempt and use torchvision directly.",
    )
    parser.add_argument(
        "--worksheet-md",
        type=Path,
        default=workspace_root / "Individual Activity.md",
        help="Path to worksheet markdown used for strict adherence checks.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate worksheet adherence and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    ensure_dir(args.data_dir)
    ensure_dir(args.output.parent)
    ensure_dir(args.artifacts_dir)

    run_artifacts_root = args.artifacts_dir / "runs"
    round_tables_dir = args.artifacts_dir / "round_tables"
    ensure_dir(run_artifacts_root)
    ensure_dir(round_tables_dir)

    rounds = build_rounds()
    worksheet_check = assert_strict_worksheet_adherence(rounds, args.worksheet_md)
    print("[adherence] Worksheet validation passed.")
    print(f"[adherence] Source worksheet: {worksheet_check['worksheet_path']}")

    if args.validate_only:
        print("[adherence] --validate-only set. Exiting without running training.")
        return

    device, device_info = select_best_device()
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("[device] No CUDA/MPS backend available. Falling back to CPU.")
    elif device.type == "cuda":
        selected_gpu = device_info.get("selected", {}).get("name", "GPU")
        print(f"[device] Using dedicated GPU: {selected_gpu}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")

    prefer_kaggle = args.prefer_kaggle and not args.skip_kaggle
    dataset_bundle = load_datasets(args.data_dir, prefer_kaggle=prefer_kaggle)
    print(f"Dataset source: {dataset_bundle.source}")
    for note in dataset_bundle.notes:
        print(f"[dataset] {note}")

    print(f"Train samples: {len(dataset_bundle.train_plain):,}")
    print(f"Test samples : {len(dataset_bundle.test_set):,}")

    round_order = ["round1", "round2", "round3", "round4", "round5", "round6", "round7"]

    all_results: list[dict[str, Any]] = []
    round_map: dict[str, list[str]] = {}
    logs_by_run: dict[str, list[str]] = {}
    picture_items: list[dict[str, str]] = []

    round7_labels: np.ndarray | None = None
    round7_preds: np.ndarray | None = None

    total_epochs = sum(cfg.epochs for r in round_order for cfg in rounds[r])
    overall_pbar = tqdm(
        total=total_epochs,
        desc="Overall Training Progress",
        unit="epoch",
        dynamic_ncols=True,
    )

    for round_name in round_order:
        overall_pbar.set_description(f"{round_name.upper()} Progress")

        run_ids: list[str] = []
        this_round_results: list[dict[str, Any]] = []

        for cfg in rounds[round_name]:
            result, model, test_loader, run_logs = run_one_config(
                cfg=cfg,
                bundle=dataset_bundle,
                device=device,
                seed=args.seed,
                overall_pbar=overall_pbar,
            )

            run_dir = run_artifacts_root / round_name / f"run_{cfg.run_id}"
            result["artifacts"] = save_run_artifacts(result, run_logs, run_dir)

            picture_items.append(
                {
                    "type": "run_curve",
                    "round": round_name,
                    "run_id": cfg.run_id,
                    "path": result["artifacts"]["curves_png"],
                }
            )
            picture_items.append(
                {
                    "type": "run_terminal",
                    "round": round_name,
                    "run_id": cfg.run_id,
                    "path": result["artifacts"]["terminal_png"],
                }
            )
            picture_items.append(
                {
                    "type": "run_summary",
                    "round": round_name,
                    "run_id": cfg.run_id,
                    "path": result["artifacts"]["summary_png"],
                }
            )

            logs_by_run[cfg.run_id] = run_logs
            all_results.append(result)
            this_round_results.append(result)
            run_ids.append(cfg.run_id)

            if cfg.run_id == "7":
                labels, preds = collect_predictions(model, test_loader, device)
                round7_labels = labels
                round7_preds = preds

            # Make sure CUDA memory is released between runs when available.
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        round_map[round_name] = run_ids

        round_table_path = round_tables_dir / f"{round_name}_table.png"
        save_round_table_image(round_name, this_round_results, round_table_path)
        picture_items.append(
            {
                "type": "round_table",
                "round": round_name,
                "run_id": "",
                "path": str(round_table_path.resolve()),
            }
        )
    
    overall_pbar.close()

    by_id = {item["run_id"]: item for item in all_results}
    round1_acc = float(by_id["1"]["test_acc"])
    round7_acc = float(by_id["7"]["test_acc"])

    top_confusion_rows: list[dict[str, int | str]] = []
    confusion_matrix_path = None
    confusion_csv_path = None
    top_confusions_png_path = None

    if round7_labels is not None and round7_preds is not None:
        top_confusion_rows = top_confusions(round7_labels, round7_preds, CLASS_NAMES, top_k=10)
        cm = confusion_matrix_array(round7_labels, round7_preds, len(CLASS_NAMES))

        confusion_matrix_path = args.artifacts_dir / "round7_confusion_matrix.png"
        save_confusion_matrix_plot(cm, CLASS_NAMES, confusion_matrix_path)
        picture_items.append(
            {
                "type": "round7_confusion_matrix",
                "round": "round7",
                "run_id": "7",
                "path": str(confusion_matrix_path.resolve()),
            }
        )

        confusion_csv_path = args.artifacts_dir / "round7_confusion_matrix.csv"
        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(confusion_csv_path)

        top_confusions_png_path = args.artifacts_dir / "round7_top_confusions.png"
        confusion_lines = [
            f"{row['count']}x: true={row['true_class']} predicted={row['pred_class']}"
            for row in top_confusion_rows
        ]
        save_text_proof_image(
            text="\n".join(confusion_lines),
            output_path=top_confusions_png_path,
            title="Round 7 Top Confusions",
            font_size=18,
            max_chars=100,
        )
        picture_items.append(
            {
                "type": "round7_top_confusions",
                "round": "round7",
                "run_id": "7",
                "path": str(top_confusions_png_path.resolve()),
            }
        )

    payload: dict[str, Any] = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "device": str(device),
            "device_info": device_info,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "data_dir": str(args.data_dir.resolve()),
            "dataset_source": dataset_bundle.source,
            "dataset_notes": dataset_bundle.notes,
            "artifacts_dir": str(args.artifacts_dir.resolve()),
        },
        "rounds": round_map,
        "results": all_results,
        "round1_test_acc": round1_acc,
        "round7": {
            "test_acc": round7_acc,
            "top_confusions": top_confusion_rows,
            "confusion_matrix_png": str(confusion_matrix_path.resolve()) if confusion_matrix_path else None,
            "confusion_matrix_csv": str(confusion_csv_path.resolve()) if confusion_csv_path else None,
            "top_confusions_png": str(top_confusions_png_path.resolve()) if top_confusions_png_path else None,
        },
        "improvement_over_round1_pp": (round7_acc - round1_acc) * 100.0,
    }

    write_json(args.output, payload)

    summary_md_path = args.output.parent / "activity_results.md"
    write_summary_markdown(summary_md_path, payload, round_order)

    logs_md_path = args.output.parent / "activity_terminal_logs.md"
    write_terminal_logs_markdown(logs_md_path, logs_by_run)

    flat_df = flatten_results(all_results)
    flat_csv_path = args.output.parent / "activity_results_flat.csv"
    flat_df.to_csv(flat_csv_path, index=False)

    env_proof_path = args.artifacts_dir / "environment_proof.png"
    env_text = "\n".join(
        [
            f"Device selected: {payload['meta']['device']}",
            f"Torch: {payload['meta']['torch']}",
            f"Torchvision: {payload['meta']['torchvision']}",
            f"Dataset source: {payload['meta']['dataset_source']}",
            *(f"Dataset note: {note}" for note in payload['meta']['dataset_notes']),
            f"Round 1 test: {round1_acc * 100:.2f}%",
            f"Round 7 test: {round7_acc * 100:.2f}%",
            f"Improvement: {(round7_acc - round1_acc) * 100:.2f} percentage points",
        ]
    )
    save_text_proof_image(
        text=env_text,
        output_path=env_proof_path,
        title="Environment and Final Metrics Proof",
        font_size=18,
        max_chars=110,
    )
    picture_items.append(
        {
            "type": "environment_proof",
            "round": "",
            "run_id": "",
            "path": str(env_proof_path.resolve()),
        }
    )

    verify_picture_structure(picture_items)

    picture_manifest_json_path = args.artifacts_dir / "picture_manifest.json"
    write_json(
        picture_manifest_json_path,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "count": len(picture_items),
            "items": picture_items,
        },
    )

    picture_index_md_path = args.output.parent / "activity_picture_index.md"
    write_picture_index(picture_index_md_path, picture_items)

    payload["meta"]["worksheet_validation"] = worksheet_check
    payload["meta"]["worksheet_md"] = str(args.worksheet_md.resolve())
    payload["meta"]["picture_manifest_json"] = str(picture_manifest_json_path.resolve())
    payload["meta"]["picture_index_md"] = str(picture_index_md_path.resolve())
    payload["meta"]["picture_count"] = len(picture_items)

    write_json(args.output, payload)

    print("\n" + "=" * 92)
    print(f"Saved JSON results: {args.output.resolve()}")
    print(f"Saved summary markdown: {summary_md_path.resolve()}")
    print(f"Saved terminal markdown: {logs_md_path.resolve()}")
    print(f"Saved picture index markdown: {picture_index_md_path.resolve()}")
    print(f"Saved picture manifest JSON: {picture_manifest_json_path.resolve()}")
    print(f"Saved flat CSV: {flat_csv_path.resolve()}")
    print(f"Saved artifact folder: {args.artifacts_dir.resolve()}")
    print(f"Total pictures saved: {len(picture_items)}")
    print(f"Round 1 test accuracy: {round1_acc * 100:.2f}%")
    print(f"Round 7 test accuracy: {round7_acc * 100:.2f}%")
    print(f"Improvement: {(round7_acc - round1_acc) * 100:.2f} percentage points")


if __name__ == "__main__":
    main()
