import argparse
import csv
import re
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError as exc:
        raise SystemExit(
            "Install tflite_runtime on the Raspberry Pi or tensorflow on the dev machine to run TFLite evaluation."
        ) from exc


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_input_size(path: Path) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"Input size:\s*(\d+)\s*x\s*(\d+)", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse input size from {path}")
    return int(match.group(1)), int(match.group(2))


def load_labels(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def center_crop_square(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    side = min(height, width)
    start_y = (height - side) // 2
    start_x = (width - side) // 2
    return image[start_y : start_y + side, start_x : start_x + side]


def preprocess_image(path: Path, input_size: tuple[int, int]) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    crop = center_crop_square(bgr)
    resized = cv2.resize(crop, input_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image = rgb.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    return np.expand_dims(image, axis=0).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Lab 10 TFLite classifier on folders of images.")
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parent.parent / "deployment_package")
    parser.add_argument("--data-dir", type=Path, required=True, help="ImageFolder-style directory with one folder per class.")
    parser.add_argument("--output-csv", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "tables" / "tflite_eval_predictions.csv")
    args = parser.parse_args()

    model_path = args.package_dir / "model.tflite"
    labels = load_labels(args.package_dir / "labels.txt")
    input_size = parse_input_size(args.package_dir / "preprocessing.txt")

    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    rows = []
    correct = 0
    total = 0
    start = time.perf_counter()
    for true_label in labels:
        class_dir = args.data_dir / true_label
        if not class_dir.exists():
            print(f"Skipping missing class directory: {class_dir}")
            continue
        for image_path in sorted(class_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            sample = preprocess_image(image_path, input_size)
            interpreter.set_tensor(input_details[0]["index"], sample)
            infer_start = time.perf_counter()
            interpreter.invoke()
            infer_ms = (time.perf_counter() - infer_start) * 1000.0
            logits = interpreter.get_tensor(output_details[0]["index"])[0]
            probs = softmax(logits.astype(np.float32))
            pred_idx = int(np.argmax(probs))
            pred_label = labels[pred_idx]
            confidence = float(probs[pred_idx])
            total += 1
            correct += int(pred_label == true_label)
            rows.append(
                {
                    "image": str(image_path),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": f"{confidence:.6f}",
                    "inference_ms": f"{infer_ms:.3f}",
                    "correct": pred_label == true_label,
                }
            )

    elapsed = time.perf_counter() - start
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "true_label", "pred_label", "confidence", "inference_ms", "correct"])
        writer.writeheader()
        writer.writerows(rows)

    accuracy = correct / total if total else 0.0
    print(f"Evaluated {total} images in {elapsed:.2f}s")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Wrote predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
