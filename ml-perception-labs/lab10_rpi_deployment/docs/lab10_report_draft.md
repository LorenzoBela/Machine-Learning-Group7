# Lab 10 Report Draft

## System Overview

The deployed system is a Raspberry Pi edge inference device for five-class grocery image classification. A USB webcam captures an image when the push-button is pressed. The image is center-cropped, resized, normalized, and passed to a TensorFlow Lite classifier. One LED is assigned to each output class, and the LED corresponding to the predicted class lights for four seconds.

## Model Choice

The Lab 9 baseline model was MobileNetV2 trial `m1_t3`, converted from PyTorch to TensorFlow Lite. It retained the Lab 8 metric after conversion, but its held-out test accuracy was only 72.00%. Per-class behavior showed that canned goods was the strongest class, while bottled water and combo were the most error-prone.

For Lab 10 improvement, the retraining pipeline uses MobileNetV3-small with `128x128` input, CUDA training, mixed precision, stronger augmentation, and weighted sampling. The model is selected using validation macro F1 and worst-class recall so weak classes matter during checkpoint selection.

## Hardware Setup

| Role | Component | GPIO Pin (BCM) | Physical Pin | Notes |
|---|---|---:|---:|---|
| Button input | Momentary push-button | 17 | 11 | Button to GND, internal pull-up |
| Class 0 indicator | LED for Noodles | 4 | 7 | 330 ohm resistor |
| Class 1 indicator | LED for Rice | 5 | 29 | 330 ohm resistor |
| Class 2 indicator | LED for bottled water | 6 | 31 | 330 ohm resistor |
| Class 3 indicator | LED for canned goods | 12 | 32 | 330 ohm resistor |
| Class 4 indicator | LED for combo | 13 | 33 | 330 ohm resistor |

## LED Mapping Rule

The model is a classification model, so the script applies argmax to the output logits/probabilities. The class with the highest confidence lights its corresponding LED:

| Class Index | Label | GPIO |
|---:|---|---:|
| 0 | Noodles | 4 |
| 1 | Rice | 5 |
| 2 | bottled water | 6 |
| 3 | canned goods | 12 |
| 4 | combo | 13 |

## Benchmark Results

| Metric | Dev Machine Baseline | Lab 10 Retrained Model | Raspberry Pi |
|---|---:|---:|---:|
| Model file size (MB) | 8.47 | 6.11 | Fill on Pi |
| Mean inference latency (ms/sample) | 1.3041 | 1.61 | Fill on Pi |
| End-to-end response time button to LED (ms) | N/A | N/A | Fill on Pi |
| Primary task metric accuracy | 72.00% | 96.00% | Fill from test cases |
| Macro F1 | 72.69% | 96.60% | N/A |

## Test Cases

| # | Input Scenario | Expected LED Behavior | Actual LED Behavior | Confidence | Pass / Fail |
|---:|---|---|---|---:|---|
| 1 | Noodles package centered | Noodles LED | Fill | Fill | Fill |
| 2 | Rice package centered | Rice LED | Fill | Fill | Fill |
| 3 | Bottled water centered | Bottled water LED | Fill | Fill | Fill |
| 4 | Canned goods centered | Canned goods LED | Fill | Fill | Fill |
| 5 | Combo item centered | Combo LED | Fill | Fill | Fill |
| 6 | Challenging glare or cluttered background | Best matching class or documented uncertainty | Fill | Fill | Fill |

## Limitations

1. No new Pi-camera training images were collected, so domain shift remains possible between the old dataset and the live camera.
2. Reflective objects such as bottled water and canned goods can be affected by glare.
3. The center crop can remove useful context if the object is not centered before the button press.
4. The combo class is inherently ambiguous because it contains features from multiple other classes.

## Mitigations

1. Use the debug images saved by the inference script to inspect the exact crop seen by the model.
2. Reduce glare by angling the object or camera away from overhead light.
3. Use a plain background and center the object before pressing the button.
4. Use macro F1 and worst-class recall during model selection instead of accuracy alone.

## Conclusion

The Lab 10 system completes the full edge inference loop: camera capture, preprocessing, TensorFlow Lite inference, and GPIO LED output. The original model was deployment-ready but not accurate enough for reliable live use. The Lab 10 retraining package improves the model using GPU training, stronger augmentation, weighted sampling, and per-class evaluation, while the Pi deployment scripts preserve the exact preprocessing contract required for accuracy retention.

