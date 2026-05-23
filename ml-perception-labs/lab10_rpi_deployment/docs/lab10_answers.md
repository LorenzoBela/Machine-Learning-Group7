# Lab 10 Reflection Answers

## 1. Why is `tflite_runtime` preferred over full TensorFlow on Raspberry Pi?

`tflite_runtime` is preferred because it contains only the inference interpreter and the kernels needed to execute `.tflite` models. It is much smaller than full TensorFlow, has fewer dependencies, installs more cleanly on ARM Linux, and avoids training-time features that are unnecessary on the Pi. This reduces storage use, setup failures, and runtime overhead.

## 2. How was one LED and the push-button wired?

Each LED is wired from a GPIO output pin through a 330 ohm resistor to the LED anode, with the LED cathode connected to ground. The resistor limits current so the LED and GPIO pin are protected. The push-button connects GPIO 17 to ground; the software uses the Pi internal pull-up resistor, so no external button resistor is required.

## 3. How does LED-output mapping differ across classification, detection, and segmentation?

For this group, the task is classification. The model outputs one vector of class logits/probabilities, and the script lights the single LED corresponding to the highest-confidence class. In object detection, multiple LEDs may light because multiple object classes can be detected in one frame. In segmentation, LEDs can represent the dominant pixel class or every class above a pixel-area threshold.

## 4. Why is the first inference slower, and how was it handled?

The first inference is slower because the TFLite interpreter allocates tensors, prepares kernels, and initializes runtime paths. The script performs a dummy warm-up inference at startup, before the first button press, so the user-facing button-to-LED path measures steady-state performance instead of cold-start overhead.

## 5. Why is Pi latency different from development-machine latency?

The Raspberry Pi runs inference on an ARM CPU with limited memory bandwidth and no desktop GPU acceleration. The development machine can use faster CPU/GPU resources, larger caches, and optimized libraries. The biggest design choice affecting Pi latency is input size: larger images improve accuracy but increase preprocessing and inference time.

