import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import ImageFilter
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v3_small


CLASS_ORDER = ["Noodles", "Rice", "bottled water", "canned goods", "combo"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEED = 42


class GaussianBlurPIL:
    def __init__(self, radius: float):
        self.radius = radius

    def __call__(self, image):
        return image.filter(ImageFilter.GaussianBlur(radius=self.radius))


def build_model(num_classes: int) -> nn.Module:
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def lab8_test_indices(dataset_len: int) -> list[int]:
    indices = torch.randperm(dataset_len, generator=torch.Generator().manual_seed(SEED)).tolist()
    n_train = int(0.60 * dataset_len)
    n_val = int(0.20 * dataset_len)
    return indices[n_train + n_val :]


def eval_transform(img_size: int, scenario: str):
    base = []
    if scenario == "baseline":
        base.append(transforms.Resize((img_size, img_size)))
    elif scenario == "center_crop_80":
        base.extend([transforms.Resize((img_size, img_size)), transforms.CenterCrop(int(img_size * 0.80)), transforms.Resize((img_size, img_size))])
    elif scenario == "brighter":
        base.extend([transforms.Resize((img_size, img_size)), transforms.ColorJitter(brightness=(1.35, 1.35))])
    elif scenario == "darker":
        base.extend([transforms.Resize((img_size, img_size)), transforms.ColorJitter(brightness=(0.65, 0.65))])
    elif scenario == "low_contrast":
        base.extend([transforms.Resize((img_size, img_size)), transforms.ColorJitter(contrast=(0.60, 0.60))])
    elif scenario == "blur":
        base.extend([transforms.Resize((img_size, img_size)), GaussianBlurPIL(radius=1.2)])
    else:
        raise ValueError(f"Unknown robustness scenario: {scenario}")
    base.extend([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
    return transforms.Compose(base)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    recalls = recall_score(y_true, y_pred, average=None, labels=list(range(len(CLASS_ORDER))), zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "worst_class_recall": float(np.min(recalls)),
        **{f"recall_{label}": float(recalls[idx]) for idx, label in enumerate(CLASS_ORDER)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Lab 10 checkpoint under simple deployment-like shifts.")
    parser.add_argument("--checkpoint", type=Path, default=Path("ml-perception-labs/lab10_rpi_deployment/deployment_package/lab10_mobilenetv3_small_best.pth"))
    parser.add_argument("--data-dir", type=Path, default=Path("ml-perception-labs/lab08_finetuning/data/raw"))
    parser.add_argument("--lab10-root", type=Path, default=Path("ml-perception-labs/lab10_rpi_deployment"))
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(CLASS_ORDER)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    temp_dataset = ImageFolder(args.data_dir)
    if temp_dataset.classes != CLASS_ORDER:
        raise RuntimeError(f"Class order mismatch. Expected {CLASS_ORDER}, got {temp_dataset.classes}")
    test_indices = lab8_test_indices(len(temp_dataset))

    rows = []
    for scenario in ["baseline", "center_crop_80", "brighter", "darker", "low_contrast", "blur"]:
        dataset = ImageFolder(args.data_dir, transform=eval_transform(args.img_size, scenario))
        loader = DataLoader(Subset(dataset, test_indices), batch_size=args.batch_size, shuffle=False, num_workers=0)
        metrics = evaluate(model, loader, device)
        rows.append({"scenario": scenario, **metrics})

    tables_dir = args.lab10_root / "outputs" / "tables"
    logs_dir = args.lab10_root / "outputs" / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = tables_dir / "lab10_robustness_eval.csv"
    df.to_csv(csv_path, index=False)
    (logs_dir / "lab10_robustness_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
