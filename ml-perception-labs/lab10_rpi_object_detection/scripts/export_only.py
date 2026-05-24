import os
import shutil
import subprocess
from pathlib import Path
from ultralytics import YOLO

# Environment setup
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Paths
WORKSPACE = Path(__file__).resolve().parents[3]  # Elective Machine Learning
LAB10_ROOT = WORKSPACE / "ml-perception-labs" / "lab10_rpi_object_detection"
OUTPUTS_DIR = LAB10_ROOT / "outputs"
BEST_PT = OUTPUTS_DIR / "yolo_runs" / "lab10_yolov8n" / "weights" / "best.pt"
DEPLOYMENT_DIR = LAB10_ROOT / "deployment_package"

CLASS_NAMES = ["Bottled Water", "Noodles", "canned-goods", "rice"]
IMG_SIZE = 640


def main():
    print("=" * 60)
    # 1. Check disk space
    free_bytes = shutil.disk_usage(WORKSPACE).free
    free_mb = free_bytes / 1024 / 1024
    print(f"Checking disk space... {free_mb:.2f} MB free.")
    if free_mb < 300:
        print("\nWARNING: You have very low disk space. Please free up at least 300MB-500MB on your C: drive before proceeding.")
        # We will still try to run, but print warning
    
    if not BEST_PT.exists():
        print(f"Error: {BEST_PT} not found. Please train the model first.")
        return

    # 2. Export to ONNX (this always succeeds and creates static shapes)
    print("\n[1/3] Exporting PyTorch model to ONNX...")
    model = YOLO(str(BEST_PT))
    onnx_path_str = model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        dynamic=False,
        simplify=True,
    )
    onnx_path = Path(onnx_path_str)
    print(f"ONNX model saved at: {onnx_path}")

    # 3. Call onnx2tf with static shape override to prevent dimension collapse crashes
    print("\n[2/3] Converting ONNX to TFLite (via onnx2tf with static shape override)...")
    cmd = [
        str(WORKSPACE / ".venv" / "Scripts" / "onnx2tf.exe"),
        "-i", str(onnx_path),
        "-ois", f"images:1,3,{IMG_SIZE},{IMG_SIZE}",
    ]
    
    # Run the onnx2tf tool
    result = subprocess.run(cmd, cwd=str(WORKSPACE))
    
    if result.returncode != 0:
        print("\nERROR: onnx2tf conversion failed. Check the logs above.")
        print("This is usually due to insufficient disk space (out of space when writing files).")
        return

    # 4. Copy the generated TFLite model to deployment package
    print("\n[3/3] Organizing deployment package...")
    tflite_src = WORKSPACE / "saved_model" / "best_float32.tflite"
    if not tflite_src.exists():
        # Fallback search in parent directory structure
        tflite_src = onnx_path.parent / "best_saved_model" / "best_float32.tflite"
    if not tflite_src.exists():
        # General rglob search in workspace saved_model
        for candidate in (WORKSPACE / "saved_model").rglob("*.tflite"):
            tflite_src = candidate
            break

    if not tflite_src.exists():
        print(f"Error: Could not locate converted TFLite model at {tflite_src}")
        return

    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
    tflite_dst = DEPLOYMENT_DIR / "model.tflite"
    shutil.copy2(tflite_src, tflite_dst)
    
    # Write metadata files
    (DEPLOYMENT_DIR / "labels.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
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
        "Deployment note: use TFLite model with NMS post-processing on Pi.",
    ])
    (DEPLOYMENT_DIR / "model_card.md").write_text(model_card + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print("SUCCESS! Deployment package ready at:")
    print(f"  {DEPLOYMENT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
