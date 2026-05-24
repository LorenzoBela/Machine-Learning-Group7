"""
Lab 10 - Object Detection Pipeline (YOLOv8n)
=============================================
Train YOLOv8-nano on the 4-class grocery detection dataset,
evaluate mAP@50, and export to TFLite for Raspberry Pi deployment.

Classes: Bottled Water, Noodles, canned-goods, rice
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import random
import shutil
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# ── Configuration ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths
WORKSPACE = Path(__file__).resolve().parents[3]  # Elective Machine Learning
RAW_DATA = WORKSPACE / "ml_project_yolov8_data"
LAB10_ROOT = WORKSPACE / "ml-perception-labs" / "lab10_rpi_object_detection"
DATASET_DIR = LAB10_ROOT / "yolo_dataset"
DEPLOYMENT_DIR = LAB10_ROOT / "deployment_package"
OUTPUTS_DIR = LAB10_ROOT / "outputs"

CLASS_NAMES = ["Bottled Water", "Noodles", "canned-goods", "rice"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 640
EPOCHS = 50
BATCH_SIZE = 16
CONF_THRESHOLD = 0.25


# ── Step 1: Split dataset into train / val / test (70/15/15) ────────────────
def split_dataset():
    """Split the train-only Roboflow export into train/val/test."""
    src_images = RAW_DATA / "train" / "images"
    src_labels = RAW_DATA / "train" / "labels"

    all_images = sorted(src_images.glob("*"))
    all_images = [p for p in all_images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    random.shuffle(all_images)

    n = len(all_images)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train:n_train + n_val],
        "test": all_images[n_train + n_val:],
    }

    for split_name, image_list in splits.items():
        img_dir = DATASET_DIR / split_name / "images"
        lbl_dir = DATASET_DIR / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in image_list:
            shutil.copy2(img_path, img_dir / img_path.name)
            label_path = src_labels / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.copy2(label_path, lbl_dir / label_path.name)

        print(f"  {split_name}: {len(image_list)} images")

    # Write data.yaml for YOLO training
    data_yaml = DATASET_DIR / "data.yaml"
    data_yaml.write_text(
        f"path: {DATASET_DIR.as_posix()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"\n"
        f"nc: {NUM_CLASSES}\n"
        f"names: {CLASS_NAMES}\n",
        encoding="utf-8",
    )
    print(f"  Wrote {data_yaml}")
    return data_yaml


# ── Step 2: Train YOLOv8-nano ───────────────────────────────────────────────
def train_model(data_yaml: Path) -> Path:
    """Train YOLOv8n and return the path to the best weights."""
    model = YOLO("yolov8n.pt")  # nano variant for Pi deployment

    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        seed=SEED,
        device=0,  # CUDA GPU
        workers=0,  # Use 0 workers on Windows to prevent multiprocessing/OpenBLAS errors
        project=str(OUTPUTS_DIR / "yolo_runs"),
        name="lab10_yolov8n",
        exist_ok=True,
        patience=10,
        save=True,
        plots=True,
        verbose=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nBest weights saved to: {best_weights}")
    return best_weights


# ── Step 3: Evaluate on test set ────────────────────────────────────────────
def evaluate_model(weights_path: Path, data_yaml: Path) -> dict:
    """Evaluate the trained model on the test split and return metrics."""
    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        conf=CONF_THRESHOLD,
        plots=True,
        save_json=True,
    )

    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    # Per-class mAP50
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(metrics.box.ap50):
            results[f"mAP50_{class_name}"] = float(metrics.box.ap50[i])

    print("\n=== Test Set Evaluation ===")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

    return results


# ── Step 4: Export to TFLite ────────────────────────────────────────────────
def export_tflite(weights_path: Path) -> Path:
    """Export the YOLOv8 model to TFLite format for Raspberry Pi."""
    model = YOLO(str(weights_path))

    model.export(
        format="tflite",
        imgsz=IMG_SIZE,
        half=False,  # keep float32 for accuracy retention
    )

    # Find the exported TFLite file
    tflite_src = weights_path.parent / "best_saved_model" / "best_float32.tflite"
    if not tflite_src.exists():
        # Alternative path structure
        export_dir = weights_path.with_suffix("")
        for candidate in export_dir.rglob("*.tflite"):
            tflite_src = candidate
            break

    tflite_dst = DEPLOYMENT_DIR / "model.tflite"
    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tflite_src, tflite_dst)
    print(f"\nTFLite model copied to: {tflite_dst}")
    print(f"  Size: {tflite_dst.stat().st_size / 1024 / 1024:.2f} MB")
    return tflite_dst


# ── Step 5: Write deployment metadata ───────────────────────────────────────
def write_deployment_files(eval_results: dict):
    """Write labels.txt, preprocessing.txt, and model_card.md."""
    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

    # labels.txt
    (DEPLOYMENT_DIR / "labels.txt").write_text(
        "\n".join(CLASS_NAMES) + "\n",
        encoding="utf-8",
    )

    # preprocessing.txt
    (DEPLOYMENT_DIR / "preprocessing.txt").write_text(
        f"Input size: {IMG_SIZE}x{IMG_SIZE} RGB\n"
        f"Model: YOLOv8n object detection\n"
        f"Input tensor: NHWC float32, shape (1, {IMG_SIZE}, {IMG_SIZE}, 3)\n"
        f"Resize: letterbox resize preserving aspect ratio\n"
        f"Color order: RGB\n"
        f"Scale: uint8 pixels to [0, 1]\n"
        f"No mean/std normalization (YOLOv8 uses 0-1 scaling only)\n",
        encoding="utf-8",
    )

    # model_card.md
    map50 = eval_results.get("mAP50", 0.0)
    model_card = "\n".join([
        "# Lab 10 YOLOv8n Object Detection Model Card",
        "",
        "Source model: YOLOv8-nano fine-tuned on grocery dataset",
        "",
        "Dataset: Roboflow ML project v5 (4-class grocery detection)",
        "",
        "Task type: 4-class object detection",
        "",
        "Classes: " + ", ".join(CLASS_NAMES),
        "",
        f"Input shape: RGB {IMG_SIZE}x{IMG_SIZE}, scaled to [0, 1]",
        "",
        "Output format: bounding boxes with class index and confidence",
        "",
        f"Test mAP@50: {map50:.4f}",
        "",
        "Deployment note: use TFLite model with NMS post-processing on Pi.",
    ])
    (DEPLOYMENT_DIR / "model_card.md").write_text(model_card + "\n", encoding="utf-8")

    print("Wrote labels.txt, preprocessing.txt, model_card.md")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Lab 10 - YOLOv8n Object Detection Pipeline")
    print("=" * 60)

    print("\n[1/5] Splitting dataset into train/val/test...")
    data_yaml = split_dataset()

    print("\n[2/5] Training YOLOv8n...")
    best_weights = train_model(data_yaml)

    print("\n[3/5] Evaluating on test set...")
    eval_results = evaluate_model(best_weights, data_yaml)

    print("\n[4/5] Exporting to TFLite...")
    export_tflite(best_weights)

    print("\n[5/5] Writing deployment files...")
    write_deployment_files(eval_results)

    print("\n" + "=" * 60)
    print("DONE! Deployment package ready at:")
    print(f"  {DEPLOYMENT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
