import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the Lab 10 ONNX model to float32 TFLite.")
    parser.add_argument("--onnx", type=Path, default=Path("ml-perception-labs/lab10_rpi_deployment/deployment_package/lab10_mobilenetv3_small.onnx"))
    parser.add_argument("--package-dir", type=Path, default=Path("ml-perception-labs/lab10_rpi_deployment/deployment_package"))
    parser.add_argument("--logs-dir", type=Path, default=Path("ml-perception-labs/lab10_rpi_deployment/outputs/logs"))
    args = parser.parse_args()

    saved_model_dir = args.package_dir / "saved_model_lab10"
    tflite_path = args.package_dir / "model.tflite"
    dummy_input_path = args.package_dir / "onnx2tf_dummy_input.npy"
    onnx2tf_sample_path = Path.cwd() / "calibration_image_sample_data_20x128x128x3_float32.npy"
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    np.save(dummy_input_path, np.zeros((1, 3, 128, 128), dtype=np.float32))
    np.save(onnx2tf_sample_path, np.zeros((20, 128, 128, 3), dtype=np.float32))
    cmd = [
        sys.executable,
        "-m",
        "onnx2tf",
        "-i",
        str(args.onnx),
        "-o",
        str(saved_model_dir),
        "-cind",
        "input",
        str(dummy_input_path),
        "-n",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    dummy_input_path.unlink(missing_ok=True)
    onnx2tf_sample_path.unlink(missing_ok=True)
    log_text = completed.stdout + "\n" + completed.stderr
    (args.logs_dir / "lab10_onnx2tf.log").write_text(log_text, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"onnx2tf failed; inspect {args.logs_dir / 'lab10_onnx2tf.log'}")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_path.write_bytes(converter.convert())
    print(f"Wrote {tflite_path}")


if __name__ == "__main__":
    main()
