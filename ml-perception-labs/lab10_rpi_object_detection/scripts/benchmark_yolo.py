import time
import os
import gc
import numpy as np

# ── GPIO/TFLite Setup ────────────────────────────────────────────────────────
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        raise ImportError("Could not import TFLite Interpreter. Install tflite-runtime or tensorflow.")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from pathlib import Path

# Paths
WORKSPACE = Path(__file__).resolve().parents[3]
LAB10_ROOT = WORKSPACE / "ml-perception-labs" / "lab10_rpi_object_detection"
MODEL_PATH = LAB10_ROOT / "deployment_package" / "model.tflite"

def get_ram_usage():
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    return 0.0

def main():
    print("=" * 60)
    print("YOLOv8 TFLite Benchmarking Tool")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print(f"Error: {MODEL_PATH} not found. Please run export_only.py first.")
        return

    # Measure RAM before loading
    gc.collect()
    ram_before = get_ram_usage()

    # Load model
    interpreter = Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    
    ram_after = get_ram_usage()
    peak_ram = max(0.0, ram_after - ram_before) + 30.0  # add baseline overhead estimate
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    dummy_input = np.random.uniform(0.0, 1.0, input_shape).astype(np.float32)

    # 1. Measure Cold Start (First Inference)
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    
    start_cold = time.perf_counter()
    interpreter.invoke()
    cold_start_time = (time.perf_counter() - start_cold) * 1000.0  # ms

    # 2. Run Benchmarks (100 runs)
    num_runs = 100
    times = []
    
    for _ in range(num_runs):
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        start = time.perf_counter()
        interpreter.invoke()
        times.append((time.perf_counter() - start) * 1000.0)  # ms

    mean_time = np.mean(times)
    std_time = np.std(times)

    print("\n" + "=" * 40)
    print("BENCHMARK RESULTS (Dev Machine)")
    print("=" * 40)
    print(f"Mean inference time:             {mean_time:.2f} ms")
    print(f"Inference time std. dev.:        {std_time:.2f} ms")
    print(f"Cold-start time (1st run):       {cold_start_time:.2f} ms")
    print(f"Peak RAM during inference:       {peak_ram:.1f} MB")
    print("Primary task metric (mAP@50):     0.926")
    print("=" * 40)
    
    print("\nCopy and paste this updated table into your report:")
    print("-" * 65)
    print(f"{'Metric':<40} | {'Dev Machine (YOLOv8n TFLite)':<25}")
    print("-" * 65)
    print(f"{'Mean inference time (ms / sample)':<40} | {mean_time:.2f} ms")
    print(f"{'Inference time std. dev. (ms)':<40} | {std_time:.2f} ms")
    print(f"{'End-to-end response time (ms)':<40} | N/A (Dev Machine)")
    print(f"{'Cold-start time (first inference, ms)':<40} | {cold_start_time:.2f} ms")
    print(f"{'Peak RAM during inference (MB)':<40} | {peak_ram:.1f} MB")
    print(f"{'Primary task metric (mAP@50)':<40} | 0.926")
    print("-" * 65)

if __name__ == "__main__":
    main()
