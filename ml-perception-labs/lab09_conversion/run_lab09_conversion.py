import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v2


NAME = "Lorenzo Bela, Robert Callorina, Kean Guzon"
SECTION = "58036"
DATE = "05/12/2026"
DATASET_NAME = "Lab04 EDA Bias Dataset (bottled water, canned goods, combo, Noodles, Rice)"
MODEL_NAME = "MobileNetV2 trial m1_t3"
SOURCE_FRAMEWORK = "PyTorch"
TASK_TYPE = "classification"
IMG_SIZE = 64
SEED = 42
LAB8_TEST_ACCURACY = 0.72
LAB8_F1_MACRO = 0.7269029315540945
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
LABS_ROOT = WORKSPACE_ROOT / "ml-perception-labs"
PROJECT_ROOT = LABS_ROOT / "lab09_conversion"
LAB8_ROOT = LABS_ROOT / "lab08_finetuning"
DATASET_ROOT = LAB8_ROOT / "data" / "raw"
LAB8_WEIGHTS = LAB8_ROOT / "finetuned_models" / "MobileNetV2_m1_t3.pth"

SOURCE_MODEL_DIR = PROJECT_ROOT / "source_model"
SAVED_MODEL_DIR = PROJECT_ROOT / "saved_model"
TFLITE_DIR = PROJECT_ROOT / "tflite_model"
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment_package"
NOTEBOOK_DIR = PROJECT_ROOT / "notebook"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

SOURCE_WEIGHTS = SOURCE_MODEL_DIR / "MobileNetV2_m1_t3.pth"
ONNX_PATH = SOURCE_MODEL_DIR / "MobileNetV2_m1_t3.onnx"
TFLITE_PATH = TFLITE_DIR / "model.tflite"
COMPARISON_CSV = DEPLOYMENT_DIR / "conversion_comparison.csv"
NOTEBOOK_PATH = NOTEBOOK_DIR / "Lab09_Conversion.ipynb"
REPORT_PATH = PROJECT_ROOT / "lab09_results_and_discussion.md"


class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = mobilenet_v2(weights=None)
        in_features = self.net.classifier[1].in_features
        self.net.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def ensure_dirs() -> None:
    for path in [
        SOURCE_MODEL_DIR,
        SAVED_MODEL_DIR,
        TFLITE_DIR,
        DEPLOYMENT_DIR,
        NOTEBOOK_DIR,
        FIGURES_DIR,
        TABLES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_dataset():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Missing Lab 8 dataset: {DATASET_ROOT}")

    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )
    dataset = ImageFolder(root=DATASET_ROOT, transform=eval_transform)
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    n_train = int(0.60 * len(indices))
    n_val = int(0.20 * len(indices))
    test_indices = indices[n_train + n_val :]
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=0)
    return dataset, test_indices, test_loader


def load_model(num_classes: int) -> nn.Module:
    if not LAB8_WEIGHTS.exists():
        raise FileNotFoundError(f"Missing Lab 8 weights: {LAB8_WEIGHTS}")
    model = MobileNetV2Classifier(num_classes)
    state_dict = torch.load(LAB8_WEIGHTS, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def collect_pytorch_outputs(model: nn.Module, loader: DataLoader):
    logits_list = []
    labels_list = []
    images_list = []
    for images, labels in loader:
        logits = model(images)
        logits_list.append(logits.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
        images_list.append(images.cpu().numpy())
    return np.concatenate(images_list), np.concatenate(labels_list), np.concatenate(logits_list)


def export_onnx(model: nn.Module) -> None:
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
    )


def convert_onnx_to_saved_model() -> str:
    if SAVED_MODEL_DIR.exists():
        shutil.rmtree(SAVED_MODEL_DIR)
    calibration_stub = PROJECT_ROOT / "calibration_image_sample_data_20x128x128x3_float32.npy"
    np.save(calibration_stub, np.zeros((20, 128, 128, 3), dtype=np.float32))
    cmd = [
        sys.executable,
        "-m",
        "onnx2tf",
        "-i",
        str(ONNX_PATH),
        "-o",
        str(SAVED_MODEL_DIR),
        "-n",
    ]
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
    calibration_stub.unlink(missing_ok=True)
    log = completed.stdout + "\n" + completed.stderr
    (PROJECT_ROOT / "onnx2tf_conversion.log").write_text(log, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"onnx2tf failed. See {PROJECT_ROOT / 'onnx2tf_conversion.log'}")
    return log


def convert_saved_model_to_tflite() -> str:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVED_MODEL_DIR))
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()
    TFLITE_PATH.write_bytes(tflite_model)
    shutil.copy2(TFLITE_PATH, DEPLOYMENT_DIR / "model.tflite")
    return "Converted SavedModel to float32 TensorFlow Lite with no optimization flags."


def prepare_for_tflite(image_chw: np.ndarray, input_shape: list[int]) -> np.ndarray:
    if len(input_shape) != 4:
        raise ValueError(f"Unsupported TFLite input shape: {input_shape}")
    if input_shape[1] == 3:
        return image_chw[np.newaxis, ...].astype(np.float32)
    return np.transpose(image_chw, (1, 2, 0))[np.newaxis, ...].astype(np.float32)


def run_tflite(images: np.ndarray):
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = [int(v) for v in input_details[0]["shape"]]

    outputs = []
    start = time.perf_counter()
    for image in images:
        sample = prepare_for_tflite(image, input_shape)
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        outputs.append(interpreter.get_tensor(output_details[0]["index"])[0])
    elapsed = time.perf_counter() - start
    latency_ms = (elapsed / len(images)) * 1000.0
    return np.asarray(outputs), latency_ms, input_details, output_details


def write_deployment_files(dataset: ImageFolder, test_indices: list[int]) -> Path:
    (DEPLOYMENT_DIR / "labels.txt").write_text("\n".join(dataset.classes) + "\n", encoding="utf-8")
    preprocessing = (
        "Input size: 64x64 RGB\n"
        "Input tensor for original PyTorch model: NCHW float32, shape (1, 3, 64, 64)\n"
        "Input tensor for converted TFLite model: inspect interpreter input; this run accepts normalized float32 image data.\n"
        "Resize: torchvision/PIL resize to 64x64\n"
        "Color order: RGB\n"
        "Scale: ToTensor converts uint8 pixels to [0, 1]\n"
        "Normalization mean: [0.485, 0.456, 0.406]\n"
        "Normalization std: [0.229, 0.224, 0.225]\n"
    )
    (DEPLOYMENT_DIR / "preprocessing.txt").write_text(preprocessing, encoding="utf-8")

    sample_source = Path(dataset.samples[test_indices[0]][0])
    sample_target = DEPLOYMENT_DIR / "sample_input.jpg"
    Image.open(sample_source).convert("RGB").save(sample_target)
    return sample_target


def save_prediction_examples(
    dataset: ImageFolder,
    test_indices: list[int],
    torch_preds: np.ndarray,
    tflite_preds: np.ndarray,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.ravel()
    for plot_idx in range(4):
        dataset_idx = test_indices[plot_idx]
        image_path, label_idx = dataset.samples[dataset_idx]
        image = Image.open(image_path).convert("RGB")
        axes[plot_idx].imshow(image)
        axes[plot_idx].axis("off")
        axes[plot_idx].set_title(
            "True: {true}\nOriginal: {orig} | TFLite: {lite}".format(
                true=dataset.classes[label_idx],
                orig=dataset.classes[int(torch_preds[plot_idx])],
                lite=dataset.classes[int(tflite_preds[plot_idx])],
            ),
            fontsize=9,
        )
    fig.tight_layout()
    output_path = FIGURES_DIR / "lab09_original_vs_tflite_examples.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def write_comparison_csv(row: dict[str, str]) -> None:
    with COMPARISON_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Metric", "Original Model (Lab 8)", "Converted TFLite Model"])
        writer.writeheader()
        for metric, values in row.items():
            writer.writerow(
                {
                    "Metric": metric,
                    "Original Model (Lab 8)": values[0],
                    "Converted TFLite Model": values[1],
                }
            )


def write_model_card(metrics: dict[str, float], input_details, output_details) -> None:
    card = f"""# MobileNetV2 m1_t3 Model Card

Source model: Lab 8 fine-tuned PyTorch MobileNetV2 trial m1_t3

Dataset: {DATASET_NAME}

Task type: 5-class image classification

Classes: bottled water, canned goods, combo, Noodles, Rice

Input shape: original PyTorch `(1, 3, 64, 64)` normalized RGB; converted TFLite input `{input_details[0]['shape'].tolist()}`

Output format: logits over 5 classes; converted TFLite output `{output_details[0]['shape'].tolist()}`

Expected test metric: Lab 8 original test accuracy {metrics['torch_accuracy']:.2%}; converted TFLite test accuracy {metrics['tflite_accuracy']:.2%}

Conversion pathway: PyTorch checkpoint -> ONNX -> TensorFlow SavedModel -> TensorFlow Lite float32
"""
    (DEPLOYMENT_DIR / "model_card.md").write_text(card, encoding="utf-8")


def write_report(metrics: dict[str, float], example_path: Path, input_details, output_details, flexops: str) -> None:
    abs_drop = metrics["torch_accuracy"] - metrics["tflite_accuracy"]
    rel_drop = abs_drop / metrics["torch_accuracy"] if metrics["torch_accuracy"] else 0.0
    faithful = (
        metrics["prediction_agreement"] >= 0.99
        and abs_drop <= 0.01
        and metrics["max_abs_error"] < 1e-3
    )
    report = f"""# Lab 09 Conversion: Results, Discussion, and Conclusion

## 1. Results Summary

Lab 9 carried over the Lab 8 deployment recommendation: **MobileNetV2 trial m1_t3** trained on the same 5-class grocery dataset. The dataset remained at 100 images per class, and the same seed-42 60/20/20 split produced 300 training samples, 100 validation samples, and 100 held-out test samples. The source checkpoint was copied from `lab08_finetuning/finetuned_models/MobileNetV2_m1_t3.pth` into the Lab 9 `source_model/` folder before conversion.

| Metric | Original Model (Lab 8) | Converted TFLite Model |
| :--- | :--- | :--- |
| Model file size (MB) | {file_size_mb(SOURCE_WEIGHTS):.2f} | {file_size_mb(TFLITE_PATH):.2f} |
| Primary task metric (accuracy) | {metrics['torch_accuracy']:.2%} | {metrics['tflite_accuracy']:.2%} |
| Macro F1 | {metrics['torch_f1']:.2%} | {metrics['tflite_f1']:.2%} |
| Max absolute output error vs. original | 0.000000 | {metrics['max_abs_error']:.6f} |
| Prediction agreement rate (%) | 100.00% original reference | {metrics['prediction_agreement']:.2%} |
| Mean inference latency (ms / sample) | {metrics['torch_latency_ms']:.4f} | {metrics['tflite_latency_ms']:.4f} |

The converted TFLite input shape reported by the interpreter was `{input_details[0]['shape'].tolist()}`, and the output shape was `{output_details[0]['shape'].tolist()}`. The side-by-side prediction figure was saved to `{example_path.as_posix()}`.

---

## 2. Conversion Pathway Summary

The source framework was **PyTorch**, so the conversion used the required cross-framework route: PyTorch checkpoint to ONNX, ONNX to TensorFlow SavedModel, and TensorFlow SavedModel to TensorFlow Lite. The PyTorch export used a fixed dummy input of `(1, 3, 64, 64)`, ONNX opset 17, input name `input`, and output name `logits`. The ONNX model was converted to TensorFlow SavedModel using `onnx2tf`, then converted to a float32 `.tflite` file using `tf.lite.TFLiteConverter.from_saved_model` with no optimization or quantization settings.

No quantization was applied, so the output values are expected to remain close to the original PyTorch logits. The converter log was saved as `onnx2tf_conversion.log`. FlexOp note: {flexops}

---

## 3. Parity and Accuracy

The original PyTorch model reached **{metrics['torch_accuracy']:.2%}** accuracy and **{metrics['torch_f1']:.2%}** macro F1 on the Lab 8 held-out test split. The converted TFLite model reached **{metrics['tflite_accuracy']:.2%}** accuracy and **{metrics['tflite_f1']:.2%}** macro F1 on the same test samples. The absolute accuracy drop was **{abs_drop:.2%}**, which is a relative drop of **{rel_drop:.2%}** from the original model.

The maximum absolute output error across the evaluated test logits was **{metrics['max_abs_error']:.6f}**, and the prediction agreement rate was **{metrics['prediction_agreement']:.2%}**. These values show that the TFLite model {'is a faithful replacement for the original model on this test split' if faithful else 'should be reviewed before deployment because at least one parity criterion changed meaningfully'}.

---

## 4. Failure Modes and Limitations

The main conversion risk was input layout. The PyTorch model was exported with NCHW input, while TensorFlow Lite commonly reports NHWC input after conversion. The verification code handled this by inspecting the TFLite interpreter input shape and feeding normalized test images in the layout expected by the converted model.

Two limitations remain for Lab 10. First, the dataset is still small at 500 total images, so the 72% test accuracy should not be treated as production-grade robustness. Second, this conversion was verified on the development machine, not on the Raspberry Pi; Lab 10 should re-check latency, memory use, and preprocessing consistency on the actual Pi runtime.

---

## 5. Questions

**Why is it necessary to convert the Lab 8 model to TensorFlow Lite for Raspberry Pi deployment instead of running the original framework directly on the Pi?**

The Lab 8 model was trained and saved in PyTorch, which is heavier than needed for inference on a Raspberry Pi. TensorFlow Lite packages the model into a compact inference-only file and can run through a smaller runtime on ARM hardware. In this lab, the PyTorch checkpoint was {file_size_mb(SOURCE_WEIGHTS):.2f} MB and the TFLite file was {file_size_mb(TFLITE_PATH):.2f} MB, but the practical benefit is the simpler deployment dependency and edge-oriented interpreter rather than file size alone.

**Explain why a TensorFlow SavedModel is produced as an explicit intermediate even when some tools can convert from PyTorch or Keras directly to TFLite.**

The TensorFlow Lite converter is designed to consume TensorFlow graphs reliably, so the SavedModel acts as the canonical TensorFlow representation before final TFLite conversion. For this PyTorch source model, ONNX served as the bridge format and `onnx2tf` produced the SavedModel. Keeping the SavedModel as an explicit artifact also gives a recoverable checkpoint if the final TFLite conversion or Pi-side testing needs to be repeated.

**Based on your parity-test results, is the converted `.tflite` a faithful replacement for the original model? Justify your answer using the maximum absolute error and the task metric reported in Part E.**

The converted model {'is faithful on the measured Lab 8 test split' if faithful else 'is not fully faithful under the strict parity criteria used here'}. The maximum absolute output error was **{metrics['max_abs_error']:.6f}**, the prediction agreement rate was **{metrics['prediction_agreement']:.2%}**, and the TFLite accuracy was **{metrics['tflite_accuracy']:.2%}** compared with **{metrics['torch_accuracy']:.2%}** for the original model. Those measured values are the basis for accepting the converted file for Lab 10 handoff.

## 6. Conclusion

Lab 9 converted the Lab 8 recommended **MobileNetV2 trial m1_t3** model from PyTorch to TensorFlow Lite using the ONNX and TensorFlow SavedModel intermediate formats. The same Lab 8 dataset, class order, seed-42 split, 64x64 RGB resizing, and ImageNet normalization were used so that the verification matched the original experiment. The final conversion produced `saved_model/` and `tflite_model/model.tflite`, with no quantization or optimization flags applied. The parity test produced a maximum absolute output error of **{metrics['max_abs_error']:.6f}** and a prediction agreement rate of **{metrics['prediction_agreement']:.2%}**. The original model reached **{metrics['torch_accuracy']:.2%}** test accuracy, while the converted TFLite model reached **{metrics['tflite_accuracy']:.2%}**, giving an absolute drop of **{abs_drop:.2%}**. The deployment package contains `model.tflite`, `labels.txt`, `preprocessing.txt`, `model_card.md`, and `sample_input.jpg`. The main open risks for Lab 10 are Raspberry Pi runtime latency, memory use, and exact preprocessing consistency on the target device. The converted TFLite model is ready for Pi-side sanity testing using the packaged sample input.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def write_notebook(metrics: dict[str, float], example_path: Path, input_details, output_details) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"""# Laboratory Exercise 9: Conversion of the Fine-Tuned Model to TensorFlow and TensorFlow Lite

**Name:** {NAME}  
**Section:** {SECTION}  
**Date:** {DATE}  
**Dataset:** {DATASET_NAME}  
**Model carried over from Lab 8:** {MODEL_NAME}
"""
        ),
        nbf.v4.new_markdown_cell(
            f"""## Part B: Source Model and Conversion Pathway

| Item | Value |
| :--- | :--- |
| Model name (from Lab 8) | MobileNetV2 trial m1_t3 |
| Source framework | PyTorch |
| Task type | Classification |
| Input shape (channels, height, width) | 3, 64, 64 |
| Preprocessing | Resize 64x64, RGB, ToTensor scale to [0, 1], normalize mean {MEAN}, std {STD} |
| Number of classes / categories | 5 |
| Baseline test metric (from Lab 8) | Accuracy {LAB8_TEST_ACCURACY:.2%}; macro F1 {LAB8_F1_MACRO:.2%} |
| Chosen conversion pathway | PyTorch -> ONNX -> TensorFlow SavedModel -> TensorFlow Lite |
"""
        ),
        nbf.v4.new_code_cell(
            """# This notebook was generated by run_lab09_conversion.py after executing the conversion.
# Re-run from the project root with:
# .\\.venv\\Scripts\\python.exe ml-perception-labs\\lab09_conversion\\run_lab09_conversion.py"""
        ),
        nbf.v4.new_markdown_cell(
            f"""## Parts C-D: Conversion Outputs

- Source checkpoint: `source_model/MobileNetV2_m1_t3.pth`
- ONNX model: `source_model/MobileNetV2_m1_t3.onnx`
- TensorFlow SavedModel: `saved_model/`
- TensorFlow Lite model: `tflite_model/model.tflite`
- TFLite input shape: `{input_details[0]['shape'].tolist()}`
- TFLite output shape: `{output_details[0]['shape'].tolist()}`

Open `saved_model/` and `tflite_model/model.tflite` in Netron for graph screenshots.
"""
        ),
        nbf.v4.new_markdown_cell(
            f"""## Part E: Verification

| Metric | Original Model (Lab 8) | Converted TFLite Model |
| :--- | :--- | :--- |
| Model file size (MB) | {file_size_mb(SOURCE_WEIGHTS):.2f} | {file_size_mb(TFLITE_PATH):.2f} |
| Primary task metric (accuracy) | {metrics['torch_accuracy']:.2%} | {metrics['tflite_accuracy']:.2%} |
| Max absolute output error vs. original | 0.000000 | {metrics['max_abs_error']:.6f} |
| Prediction agreement rate (%) | 100.00% original reference | {metrics['prediction_agreement']:.2%} |
| Mean inference latency (ms / sample) | {metrics['torch_latency_ms']:.4f} | {metrics['tflite_latency_ms']:.4f} |

Comparison CSV saved to `deployment_package/conversion_comparison.csv`.
"""
        ),
        nbf.v4.new_markdown_cell(f"## Example Predictions\n\n![Original vs TFLite](../{example_path.relative_to(PROJECT_ROOT).as_posix()})"),
        nbf.v4.new_markdown_cell(
            f"""## Part F: Deployment Package

The deployment package contains:

- `model.tflite`
- `labels.txt`
- `preprocessing.txt`
- `model_card.md`
- `sample_input.jpg`
"""
        ),
        nbf.v4.new_markdown_cell(
            "## Results and Discussion\n\nSee `../lab09_results_and_discussion.md` for the full data-driven discussion, individual question answers, and conclusion."
        ),
    ]
    nbf.write(nb, NOTEBOOK_PATH)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ensure_dirs()

    dataset, test_indices, test_loader = load_dataset()
    if dataset.classes != ["Noodles", "Rice", "bottled water", "canned goods", "combo"]:
        print(f"Detected ImageFolder class order: {dataset.classes}")

    shutil.copy2(LAB8_WEIGHTS, SOURCE_WEIGHTS)
    model = load_model(len(dataset.classes))

    start = time.perf_counter()
    images, labels, torch_logits = collect_pytorch_outputs(model, test_loader)
    torch_latency_ms = ((time.perf_counter() - start) / len(labels)) * 1000.0
    torch_preds = torch_logits.argmax(axis=1)

    export_onnx(model)
    conversion_log = convert_onnx_to_saved_model()
    convert_saved_model_to_tflite()

    tflite_logits, tflite_latency_ms, input_details, output_details = run_tflite(images)
    tflite_preds = tflite_logits.argmax(axis=1)

    metrics = {
        "torch_accuracy": float(accuracy_score(labels, torch_preds)),
        "tflite_accuracy": float(accuracy_score(labels, tflite_preds)),
        "torch_f1": float(f1_score(labels, torch_preds, average="macro", zero_division=0)),
        "tflite_f1": float(f1_score(labels, tflite_preds, average="macro", zero_division=0)),
        "max_abs_error": float(np.max(np.abs(torch_logits - tflite_logits))),
        "prediction_agreement": float(np.mean(torch_preds == tflite_preds)),
        "torch_latency_ms": float(torch_latency_ms),
        "tflite_latency_ms": float(tflite_latency_ms),
    }

    comparison_rows = {
        "Model file size (MB)": (f"{file_size_mb(SOURCE_WEIGHTS):.2f}", f"{file_size_mb(TFLITE_PATH):.2f}"),
        "Primary task metric (accuracy)": (f"{metrics['torch_accuracy']:.2%}", f"{metrics['tflite_accuracy']:.2%}"),
        "Max absolute output error vs. original": ("0.000000", f"{metrics['max_abs_error']:.6f}"),
        "Prediction agreement rate (%)": ("100.00% original reference", f"{metrics['prediction_agreement']:.2%}"),
        "Mean inference latency (ms / sample)": (f"{metrics['torch_latency_ms']:.4f}", f"{metrics['tflite_latency_ms']:.4f}"),
    }
    write_comparison_csv(comparison_rows)
    sample_path = write_deployment_files(dataset, test_indices)
    example_path = save_prediction_examples(dataset, test_indices, torch_preds, tflite_preds)
    flexops = "No FlexOps were detected in the converter log." if "Flex" not in conversion_log else "Flex-related text appeared in the converter log; inspect `onnx2tf_conversion.log` before Raspberry Pi deployment."
    write_model_card(metrics, input_details, output_details)
    write_report(metrics, example_path, input_details, output_details, flexops)
    write_notebook(metrics, example_path, input_details, output_details)

    summary = {
        "classes": dataset.classes,
        "test_samples": int(len(labels)),
        "sample_input": str(sample_path),
        "metrics": metrics,
        "tflite_input_shape": input_details[0]["shape"].tolist(),
        "tflite_output_shape": output_details[0]["shape"].tolist(),
    }
    (PROJECT_ROOT / "lab09_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
