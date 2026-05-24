import re
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from gpiozero import Button, LED
    USING_GPIO = True
except (ImportError, OSError):
    USING_GPIO = False
    print("Warning: gpiozero not found or not running on Raspberry Pi. Using mock GPIO for testing.")
    
    class Button:
        def __init__(self, pin, bounce_time=0.1):
            self.pin = pin
            print(f"[Mock Button] Initialized on pin {pin}")
        def wait_for_press(self):
            input("\n[Mock Button] Press Enter key in terminal to simulate button press...")

    class LED:
        def __init__(self, pin):
            self.pin = pin
            self.state = False
        def on(self):
            if not self.state:
                self.state = True
                print(f"[Mock LED Pin {self.pin}] ON")
        def off(self):
            if self.state:
                self.state = False
                print(f"[Mock LED Pin {self.pin}] OFF")
        def close(self):
            pass

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        raise ImportError("Could not import TFLite Interpreter. Install tflite-runtime or tensorflow.")

DEPLOYMENT_DIR = Path(__file__).resolve().parent.parent / "deployment_package"
MODEL_PATH = DEPLOYMENT_DIR / "model.tflite"
LABELS_PATH = DEPLOYMENT_DIR / "labels.txt"
PREPROCESSING_PATH = DEPLOYMENT_DIR / "preprocessing.txt"
DEBUG_RAW_PATH = Path("debug_raw.jpg")
DEBUG_CROP_PATH = Path("debug_crop.jpg")

BUTTON_PIN = 17
LED_PINS = [6, 4, 12, 5]
DISPLAY_INTERVAL = 4.0
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45


def load_labels(path: Path) -> list[str]:
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ["Bottled Water", "Noodles", "canned-goods", "rice"]


def letterbox_resize(image: np.ndarray, target_size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    dx = (target_size - new_w) // 2
    dy = (target_size - new_h) // 2
    canvas[dy:dy+new_h, dx:dx+new_w] = resized
    
    return canvas, scale, (dx, dy)


def preprocess_frame(frame_bgr: np.ndarray, target_size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    canvas, scale, pad = letterbox_resize(rgb, target_size)
    image = canvas.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0).astype(np.float32), scale, pad


def turn_all_off(leds: list[LED]) -> None:
    for led in leds:
        led.off()


def capture_frame() -> np.ndarray | None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open USB webcam.")
        return None
    
    for _ in range(5):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def postprocess_yolov8(output: np.ndarray, scale: float, pad: tuple[int, int], orig_shape: tuple[int, int]) -> list[tuple[int, float, list[int]]]:
    output = np.squeeze(output)
    output = output.T
    
    boxes = []
    confidences = []
    class_ids = []
    
    orig_h, orig_w = orig_shape
    pad_x, pad_y = pad
    
    for row in output:
        scores = row[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > CONF_THRESHOLD:
            cx, cy, w, h = row[0:4]
            cx_unpad = cx - pad_x
            cy_unpad = cy - pad_y
            
            x1 = int((cx_unpad - w / 2) / scale)
            y1 = int((cy_unpad - h / 2) / scale)
            w_orig = int(w / scale)
            h_orig = int(h / scale)
            
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            w_orig = max(1, min(w_orig, orig_w - x1))
            h_orig = max(1, min(h_orig, orig_h - y1))
            
            boxes.append([x1, y1, w_orig, h_orig])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))
            
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    
    results = []
    if len(indices) > 0:
        indices = np.array(indices).flatten()
        for idx in indices:
            results.append((class_ids[idx], confidences[idx], boxes[idx]))
            
    return results


def main() -> None:
    labels = load_labels(LABELS_PATH)
    
    if not MODEL_PATH.exists():
        print(f"Error: model.tflite not found at {MODEL_PATH}.")
        print("Please run Lab10_ObjectDetection.py first to train and export the model.")
        return

    interpreter = Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    img_size = input_shape[1] if input_shape[1] != 1 and input_shape[1] != 3 else input_shape[2]

    print(f"Loaded labels: {labels}")
    print(f"Model Input Shape: {input_shape}")
    print(f"Model Output Shape: {output_details[0]['shape']}")

    dummy = np.zeros(input_shape, dtype=np.float32)
    interpreter.set_tensor(input_details[0]["index"], dummy)
    interpreter.invoke()
    print("Warm-up inference complete.")

    if len(LED_PINS) != len(labels):
        raise ValueError(f"Need one LED pin per class: {len(labels)} labels, {len(LED_PINS)} LED pins.")

    button = Button(BUTTON_PIN, bounce_time=0.1)
    leds = [LED(pin) for pin in LED_PINS]

    print("\nSystem ready. Press the button to capture and run YOLOv8 detection.")
    try:
        while True:
            button.wait_for_press()
            print("\nButton pressed. Capturing frame...")
            start_total = time.perf_counter()
            frame = capture_frame()
            if frame is None:
                print("Failed to capture image.")
                continue

            orig_h, orig_w = frame.shape[:2]
            
            input_tensor, scale, pad = preprocess_frame(frame, target_size=img_size)
            
            if input_shape[1] == 3:
                input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))
                
            interpreter.set_tensor(input_details[0]["index"], input_tensor)
            
            start_infer = time.perf_counter()
            interpreter.invoke()
            infer_ms = (time.perf_counter() - start_infer) * 1000.0
            
            raw_output = interpreter.get_tensor(output_details[0]["index"])
            
            detections = postprocess_yolov8(raw_output, scale, pad, (orig_h, orig_w))
            total_ms = (time.perf_counter() - start_total) * 1000.0

            active_class_ids = set()
            print("\nDetections:")
            if not detections:
                print("  No objects detected.")
            else:
                for class_id, conf, box in detections:
                    active_class_ids.add(class_id)
                    label = labels[class_id]
                    print(f"  - {label}: conf={conf:.2f}, box={box}")

            turn_all_off(leds)
            for class_id in active_class_ids:
                leds[class_id].on()

            print(f"\nInference: {infer_ms:.2f} ms | End-to-end: {total_ms:.2f} ms")
            print("Displaying results on LEDs...")
            
            cv2.imwrite(str(DEBUG_RAW_PATH), frame)
            
            annotated_frame = frame.copy()
            for class_id, conf, box in detections:
                x1, y1, w_box, h_box = box
                cv2.rectangle(annotated_frame, (x1, y1), (x1 + w_box, y1 + h_box), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"{labels[class_id]} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite(str(DEBUG_CROP_PATH), annotated_frame)

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
