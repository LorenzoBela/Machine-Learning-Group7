import re
import time
from pathlib import Path

import cv2
import numpy as np
from gpiozero import Button, LED

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # Allows quick checks on machines with full TensorFlow only.
    from tensorflow.lite.python.interpreter import Interpreter


DEPLOYMENT_DIR = Path(__file__).resolve().parent.parent / "deployment_package"
MODEL_PATH = DEPLOYMENT_DIR / "model.tflite"
LABELS_PATH = DEPLOYMENT_DIR / "labels.txt"
PREPROCESSING_PATH = DEPLOYMENT_DIR / "preprocessing.txt"
DEBUG_RAW_PATH = Path("/home/jhosil/lab10_deployment/debug_raw.jpg")
DEBUG_CROP_PATH = Path("/home/jhosil/lab10_deployment/debug_crop.jpg")

DEFAULT_INPUT_SIZE = (64, 64)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DISPLAY_INTERVAL = 4.0
CONFIDENCE_THRESHOLD = 0.0

BUTTON_PIN = 17
LED_PINS = [4, 5, 6, 12, 13]


def load_labels(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_input_size(path: Path) -> tuple[int, int]:
    if not path.exists():
        return DEFAULT_INPUT_SIZE
    text = path.read_text(encoding="utf-8")
    match = re.search(r"Input size:\s*(\d+)\s*x\s*(\d+)", text, flags=re.IGNORECASE)
    if not match:
        return DEFAULT_INPUT_SIZE
    width = int(match.group(1))
    height = int(match.group(2))
    return width, height


def center_crop_square(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    side = min(height, width)
    start_y = (height - side) // 2
    start_x = (width - side) // 2
    return image[start_y : start_y + side, start_x : start_x + side]


def preprocess_frame(frame_bgr: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    cropped = center_crop_square(frame_bgr)
    resized = cv2.resize(cropped, input_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image = rgb.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    return np.expand_dims(image, axis=0).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def turn_all_off(leds: list[LED]) -> None:
    for led in leds:
        led.off()


def capture_frame() -> np.ndarray | None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open USB webcam.")
        return None
    for _ in range(15):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main() -> None:
    labels = load_labels(LABELS_PATH)
    input_size = parse_input_size(PREPROCESSING_PATH)
    interpreter = Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Loaded labels: {labels}")
    print(f"Using input size: {input_size}")
    print(f"Model input: {input_details[0]['shape']} {input_details[0]['dtype']}")
    print(f"Model output: {output_details[0]['shape']} {output_details[0]['dtype']}")

    dummy = np.zeros(input_details[0]["shape"], dtype=np.float32)
    interpreter.set_tensor(input_details[0]["index"], dummy)
    interpreter.invoke()
    print("Warm-up inference complete.")

    if len(LED_PINS) != len(labels):
        raise ValueError(f"Need one LED pin per class: {len(labels)} labels, {len(LED_PINS)} LED pins.")

    button = Button(BUTTON_PIN, bounce_time=0.1)
    leds = [LED(pin) for pin in LED_PINS]

    print("System ready. Press the button to capture and infer.")
    try:
        while True:
            button.wait_for_press()
            print("\nButton pressed. Capturing frame...")
            start_total = time.perf_counter()
            frame = capture_frame()
            if frame is None:
                print("Failed to capture image.")
                continue

            brightness = float(np.mean(frame))
            glare_pct = float(np.sum(frame > 240) / frame.size * 100.0)
            print(f"Average brightness: {brightness:.2f}; glare pixels >240: {glare_pct:.1f}%")
            if glare_pct > 8.0:
                print("Warning: strong glare detected. Angle the item or camera away from overhead light.")

            crop = center_crop_square(frame)
            cv2.imwrite(str(DEBUG_RAW_PATH), frame)
            cv2.imwrite(str(DEBUG_CROP_PATH), crop)

            input_tensor = preprocess_frame(frame, input_size)
            interpreter.set_tensor(input_details[0]["index"], input_tensor)
            start_infer = time.perf_counter()
            interpreter.invoke()
            infer_ms = (time.perf_counter() - start_infer) * 1000.0
            total_ms = (time.perf_counter() - start_total) * 1000.0

            logits = interpreter.get_tensor(output_details[0]["index"])[0]
            probs = softmax(logits.astype(np.float32))
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            pred_label = labels[pred_idx]

            turn_all_off(leds)
            if confidence >= CONFIDENCE_THRESHOLD:
                leds[pred_idx].on()
            print(
                f"Prediction: {pred_label} (class {pred_idx}) | "
                f"confidence={confidence:.4f} | inference={infer_ms:.2f} ms | end-to-end={total_ms:.2f} ms"
            )
            time.sleep(DISPLAY_INTERVAL)
            turn_all_off(leds)
            print("Ready for next capture.")
    except KeyboardInterrupt:
        print("\nExiting gracefully.")
    finally:
        turn_all_off(leds)
        for led in leds:
            led.close()
        button.close()
        print("GPIO cleanup complete.")


if __name__ == "__main__":
    main()
