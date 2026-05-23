import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter
from gpiozero import Button, LED

# Configuration
# Make sure to update these paths if your files are in a different location on your Pi
MODEL_PATH = "/home/pi/lab10_deployment/model.tflite"
LABELS_PATH = "/home/pi/lab10_deployment/labels.txt"

# Preprocessing config from Lab 9 (MobileNetV2_m1_t3)
INPUT_SIZE = (64, 64)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DISPLAY_INTERVAL = 4 # seconds LEDs stay on

# GPIO Pin Configuration (BCM numbering)
# Update these to match how you wired your breadboard!
BUTTON_PIN = 17 
# List of 5 LED pins corresponding to your 5 classes: Noodles, Rice, bottled water, canned goods, combo
LED_PINS = [4, 5, 6, 12, 13] 

def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def preprocess_image(image):
    # Resize to 64x64
    image = cv2.resize(image, INPUT_SIZE)
    # Convert BGR (OpenCV format) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Scale pixels to [0, 1]
    image = image.astype(np.float32) / 255.0
    # Normalize with mean and std
    image = (image - MEAN) / STD
    # Expand dims to match (1, 64, 64, 3) NHWC layout expected by TFLite
    image = np.expand_dims(image, axis=0)
    return image

def main():
    print("Initializing...")
    try:
        labels = load_labels(LABELS_PATH)
    except FileNotFoundError:
        print(f"Could not find {LABELS_PATH}. Make sure the path is correct.")
        labels = [f"Class {i}" for i in range(5)]
    
    # Initialize Interpreter
    try:
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
    except ValueError:
        print(f"Could not find {MODEL_PATH}. Make sure the file exists.")
        return
        
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Warm-up inference to eliminate cold-start delay
    dummy_input = np.zeros(input_details[0]['shape'], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    print("Warm-up complete.")
    
    # Initialize GPIO components
    button = Button(BUTTON_PIN, bounce_time=0.1)
    leds = [LED(pin) for pin in LED_PINS]
    
    # Initialize Camera (USB Webcam using OpenCV)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open USB webcam.")
        return
    
    print("System ready. Press the button to capture and infer.")
    
    try:
        while True:
            # Wait for button press
            button.wait_for_press()
            print("\nButton pressed! Capturing...")
            
            # Flush camera buffer to ensure a recent frame is captured
            for _ in range(5): 
                cap.read()
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to capture image.")
                continue
            
            # Preprocess
            start_time = time.time()
            input_tensor = preprocess_image(frame)
            
            # Infer
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            
            # Decode output
            output_tensor = interpreter.get_tensor(output_details[0]['index'])[0]
            predicted_class_idx = np.argmax(output_tensor)
            confidence = output_tensor[predicted_class_idx]
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            
            predicted_label = labels[predicted_class_idx]
            print(f"Prediction: {predicted_label} (Class {predicted_class_idx})")
            print(f"Confidence: {confidence:.4f} | Latency: {latency_ms:.2f} ms")
            
            # Drive the LED corresponding to the predicted class
            leds[predicted_class_idx].on()
            time.sleep(DISPLAY_INTERVAL)
            leds[predicted_class_idx].off()
            
            print("Ready for next capture.")
            
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        # Cleanup hardware resources
        cap.release()
        for led in leds:
            led.off()
        button.close()

if __name__ == "__main__":
    main()
