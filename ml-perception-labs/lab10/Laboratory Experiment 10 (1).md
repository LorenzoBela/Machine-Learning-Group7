  
**MACHINE LEARNING AND PERCEPTION LAB**

**Laboratory Exercise 10**

**Deployment of the TensorFlow Lite Model on the Raspberry Pi with Button-Triggered Capture and Class-Indicating LEDs**  
Submitted by:  
**Group** 

| Category | Exceptional 4 | Acceptable 3 | Marginal 2 | Unacceptable 1 |  Score |
| :---: | ----- | ----- | ----- | ----- | ----- |
| **System / Pipeline Design & Implementation (30%)** | Clear, well-structured machine learning pipeline or experimental design that fully meets the stated objectives, requirements, and constraints of the lab. | Adequate pipeline or experimental design with minor limitations; meets most lab requirements. | Partial or loosely structured design; some requirements addressed but key elements are missing or incorrect. | Minimal or unclear design effort; does not address the lab requirements. |  |
| **Application of Tools & Techniques (25%)** | Correct selection and expert use of appropriate tools and techniques (e.g., Python, Jupyter, ML libraries, data analysis tools); methods are effectively applied and justified. | Correct tool selection with minor errors or inconsistencies in application. | Limited, inappropriate, or incorrect tool usage; techniques partially support the task. | No meaningful or incorrect use of required tools and techniques. |  |
| **Implementation & Resource Utilization (20%)** | Efficient, logical, and well-organized implementation; methods and resources are fully aligned with the problem and constraints. | Functional implementation with minor inefficiencies or redundancies. | Implementation partially works but lacks efficiency, clarity, or completeness. | Poor or non-functional implementation with little consideration of constraints. |  |
| **Testing, Analysis & Validation  (15%)** | Comprehensive testing and analysis; results are clearly validated, interpreted, and supported by appropriate metrics, figures, or tables. | Adequate testing and analysis with mostly correct interpretation of results. | Limited testing; analysis is incomplete, weakly supported, or partially incorrect. | No testing performed or results are incorrectly analyzed or interpreted. |  |
| **Documentation & Reporting (10%)** | Clear, complete, and well-structured lab report/notebook with proper figures, tables, explanations, and reflection. | Complete documentation with minor issues in clarity, organization, or detail. | Partial documentation; missing sections, unclear explanations, or poor organization. | Incomplete, poorly written, or missing documentation. |  |
| **TOTAL SCORE** |  |  |  |  |      |

| Group Members |  |  |  |
| ----- | ----- | ----- | ----- |
| **STUDENT NUMBER** | **NAME** | **CONTRIBUTION** | **SCORE** |
|   |   |   |   |
|   |   |   |   |
|   |   |   |   |
|   |   |   |   |

 

Submitted to:  
Engr. Dexter James L. Cuaresma

Date:  
	mm/dd/year

* Set up a Raspberry Pi with the operating system, Python environment, and libraries required to run TensorFlow Lite inference on the edge (tflite\_runtime, camera driver, GPIO library).

* Wire a push-button as a capture trigger and one LED per output class to the Raspberry Pi GPIO header, with correct current-limiting resistors and a documented pin assignment.

* Deploy the .tflite model from Laboratory Exercise 9 on the Raspberry Pi and implement the full capture, preprocess, inference, output pipeline appropriate to the task type (classification, object detection, or semantic / instance segmentation).

* Map the model output to the LED array according to the task-specific rule defined in this lab, so the lit LEDs communicate the model’s prediction at a glance.

* Validate end-to-end behavior on representative inputs and benchmark on-device inference latency, end-to-end response time, and accuracy retention compared to the Lab 9 development-machine baseline.

# **Introduction**

Laboratory Exercises 6 through 9 produced a deployment-ready .tflite model. Laboratory Exercise 10 closes the loop by running that model on a Raspberry Pi and giving it a physical interface to the world. The system built in this lab has three real-world components: a camera that captures images, a push-button that triggers each capture, and a set of LEDs that visually indicate the model’s prediction. Power on the Pi, point the camera at a subject, press the button, and watch which LED lights up. The result is a small, self-contained inference appliance built end-to-end from the group’s own dataset and trained model.

Because different groups produced different kinds of models in Labs 7–9 — classifiers, object detectors, and segmentation networks — the inference loop is the same but the output stage differs by task. This lab specifies a general framework and three task-specific mappings between model output and LED behavior.

# **Detailed Discussion**

## **1\. The Raspberry Pi as an Edge Inference Device**

The Raspberry Pi is a single-board computer running a full Linux distribution on an ARM CPU. For the purposes of this lab three of its features matter: it has enough CPU and RAM to run a TensorFlow Lite model in real time, it exposes a 40-pin GPIO header that gives software-level access to digital inputs and outputs, and it has a dedicated camera interface (CSI) as well as USB ports for webcams. The result is a device that can perceive, infer, and react — all on the same board, without a network connection.

Two constraints shape the deployment. First, there is no GPU acceleration available to TFLite on standard Raspberry Pi OS; inference runs on the CPU, accelerated by the XNNPACK delegate. Second, RAM is limited (1–8 GB depending on the Pi model), so the entire model plus the largest activation tensor must fit comfortably in memory. The 64-bit version of Raspberry Pi OS is recommended for machine learning workloads because tflite\_runtime ships better-optimized kernels for aarch64.

## **2\. The TFLite Runtime on the Raspberry Pi**

The full tensorflow Python package is not the right choice on the Pi. It is large (several hundred megabytes), has many dependencies that are awkward on ARM Linux, and includes training-time machinery that is not needed for inference. Instead, the lightweight package tflite\_runtime is used. It contains only the Interpreter and the operator kernels required to execute a .tflite file, and it installs cleanly through pip on a 64-bit Raspberry Pi OS.

The Interpreter follows a simple lifecycle: load the .tflite file, allocate tensors, write the input tensor, invoke the model, and read the output tensor. The first invocation is usually slower than the rest because allocation and kernel selection happen on the first call. For a real-time system, this means the model should be loaded and warmed up at startup — not on the first button press — so the user does not see the cold-start delay.

## **3\. GPIO Fundamentals on the Raspberry Pi**

The 40-pin header on the Raspberry Pi exposes general-purpose input/output (GPIO) pins that the operating system can read from or drive at 3.3 V logic levels. Each pin can be configured as either an input (reading a digital signal from the world) or an output (driving a digital signal to the world). Pins are referenced by their BCM (Broadcom) number, which is what Python libraries use, rather than by their physical position on the header.

Two libraries are commonly used to access GPIO from Python: gpiozero, a high-level API that wraps each pin in an object (Button, LED, etc.) and is recommended for beginners; and RPi.GPIO, a lower-level API that gives finer control. This lab can be implemented with either, but gpiozero is the cleaner choice for the capture button and the class LEDs.

Three practical rules apply when wiring GPIO. First, every pin tolerates only 3.3 V — connecting a 5 V signal can damage the SoC. Second, output pins can source or sink only a small amount of current (a few milliamps), which is why current-limiting resistors are required for LEDs. Third, certain pins are reserved by default for specific peripherals (UART, I2C, SPI); the safe general-purpose pins for arbitrary use are GPIO 4, 5, 6, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, and 27\.

## **4\. The Push-Button as a Capture Trigger**

This system does not run inference continuously. Continuous inference would drain power, keep the CPU hot, and produce a stream of predictions that the LEDs cannot meaningfully display. Instead, a single button press triggers a single capture-and-infer cycle, and the LEDs hold the result until the next press.

Wiring a momentary button is straightforward: one terminal connects to the chosen GPIO pin, the other terminal connects to ground (GND). The GPIO pin is configured as an input with the Pi’s internal pull-up resistor enabled. The pin then idles at logic HIGH; pressing the button connects it to GND and drives the line to logic LOW. The software detects the falling edge as a press event.

Two practical issues arise. The first is debouncing — mechanical buttons produce dozens of fast electrical transitions over a few milliseconds when pressed. Without debouncing, one physical press can register as several events. The standard fix is to ignore further events for a short interval (typically 50–100 ms) after a detected press, either in hardware (an RC filter) or in software (a delay or the bounce-time parameter in gpiozero’s Button class). The second is responsiveness — the inference loop should not be busy when the button is pressed, or the press may be missed. Using an edge-triggered callback or wait\_for\_press in a clean main loop avoids this.

## **5\. LEDs as Class Indicators**

Each output class of the model is mapped to one physical LED. The mapping is one-to-one: class 0 → LED 0, class 1 → LED 1, and so on. Using different LED colors makes the output easier to read at a glance, especially when the dataset has semantically distinct classes (e.g., red for “defective,” green for “pass,” yellow for “maybe”).

LED wiring follows a fixed pattern: the chosen GPIO pin connects through a 330 Ω current-limiting resistor to the LED’s anode (the longer leg), and the LED’s cathode (the shorter leg) connects to ground. Writing HIGH to the GPIO pin lights the LED; writing LOW turns it off. The 330 Ω value is a safe default for standard 20 mA LEDs at 3.3 V. Without this resistor, the LED would draw too much current, dim quickly, and risk damaging the GPIO pin.

If the model has many classes (more than the safe general-purpose pins can support — roughly 16 free pins are available in practice), three options exist: (1) use addressable RGB LED strips such as WS2812 / NeoPixel, which chain many LEDs onto a single data pin; (2) use a shift register such as the 74HC595 to expand outputs; or (3) group semantically related classes onto a shared LED. This lab does not require any of these extensions if the class count is small, but groups working with large class counts should document the choice.

## **6\. Task-Specific Output Mapping**

The framework above defines what an LED does — light up to indicate a class. What it means to “indicate a class” depends on the task type. The table below specifies the rule each group must implement, based on the task type of the model carried over from Lab 9\.

| Task Type | Model Output | Recommended LED Behavior |
| :---- | :---- | :---- |
| **Classification** | A single vector of class probabilities of length N (one entry per class). | Light up the single LED corresponding to argmax of the output. Optionally, leave all LEDs off if the maximum probability is below a confidence threshold. |
| **Object Detection** | A variable-length list of detections, each containing a bounding box, a class index, and a confidence score. | After applying NMS and a confidence threshold, light up the LED corresponding to every detected class. Optionally, encode the number of detections per class as LED brightness using PWM. |
| **Semantic / Instance Segmentation** | A per-pixel class map of the same spatial size as the input image (or a set of mask \+ class predictions for instance segmentation). | Count the pixels per class. Light up the LED for the dominant class (highest pixel count), or light up every class whose pixel count exceeds a fixed fraction of the image area. |

Each group implements only the row that corresponds to its task type. The chosen rule and any thresholds (confidence threshold, pixel-fraction threshold) must be documented in the report.

## **7\. The End-to-End Inference Loop**

The full pipeline executed for each button press has six steps, all running on the Pi:

* Capture: read one frame from the camera buffer (Pi Camera via picamera2, or USB webcam via OpenCV).

* Preprocess: resize, normalize, and reorder color channels exactly according to the preprocessing.txt produced in Lab 9\. Mismatches here are the most common cause of on-device accuracy loss.

* Set input tensor: copy the preprocessed image into the Interpreter’s input buffer (set\_tensor).

* Invoke: run inference (invoke()). This is the step whose latency is measured for benchmarking.

* Read and decode output: read the output tensor(s) (get\_tensor) and apply the task-appropriate decoding — argmax for classification, non-maximum suppression for detection, or pixel-class counting for segmentation.

* Drive LEDs: write to the GPIO outputs according to the decoded result; hold the LED state for a fixed display interval (e.g., 3–5 seconds) so the human user can read it; then turn all LEDs off and return to waiting for the next button press.

## **8\. Performance Considerations**

Three performance characteristics deserve attention. First, the cold-start latency — the time from program start to the first usable inference — is dominated by model loading and tensor allocation, not by inference itself. Warming up the model with a dummy inference at startup eliminates this delay from the user-facing path. Second, the steady-state inference latency on a Raspberry Pi is typically several times slower than on the development machine used in Lab 9; the on-device measurement captured in this lab is the true latency that matters for deployment. Third, the user-perceived latency — the time from pressing the button to seeing the LED change — is the sum of camera capture, preprocessing, inference, and GPIO write times, plus any debounce delay. Keeping the model input resolution modest is the single biggest lever for shortening this end-to-end time.

**Hardware**

| Raspberry Pi 4 or 5 | 1 | 4 GB RAM minimum recommended; 64-bit Raspberry Pi OS. |
| :---- | :---: | :---- |
| **Pi Camera Module (v2 / v3) or USB webcam** | 1 | Pi Camera requires picamera2; USB webcam uses OpenCV. |
| **MicroSD card** | 1 | 32 GB minimum, Class 10 / A1. |
| **Power supply** | 1 | Official Pi power adapter (USB-C for Pi 4/5). |
| **Push button (momentary tactile)** | 1 | Used as capture trigger. |
| **LED** | N | One LED per class (N \= number of model output classes). Different colors recommended. |
| **Resistor, 330 Ω** | N | Current limiting for each LED. |
| **Resistor, 10 kΩ** | 0 or 1 | Optional pull-down for button if not using internal pull-up. |
| **Breadboard** | 1 | Solderless prototyping board. |
| **Jumper wires (M–F)** | Several | For Pi-to-breadboard connections. |

## **Software**

* Raspberry Pi OS, 64-bit (Bookworm or newer recommended).

* Python 3.10 or newer (ships with the OS).

* tflite\_runtime — the lightweight TFLite inference library.

* picamera2 (for Pi Camera Module) or opencv-python (for USB webcam).

* gpiozero — high-level GPIO library (pre-installed on Raspberry Pi OS).

* numpy and Pillow (PIL) — for preprocessing.

## **Libraries**

* tensorflow — provides tf.lite.TFLiteConverter

* onnx and onnx2tf — used when the source is PyTorch or scikit-learn

* ultralytics — used when the source is a YOLO model

* skl2onnx — used when the source is a scikit-learn model

* numpy, pillow, matplotlib

## **Dataset / Data Source**

* The final fine-tuned model weights produced in Laboratory Exercise 8

* The same dataset and the same test split used in Laboratory Exercises 6–8

# 

## **Part A) Set Up the Raspberry Pi**

* Flash Raspberry Pi OS 64-bit to a microSD card using Raspberry Pi Imager. During imaging, configure the hostname, username and password, Wi-Fi network, locale, and enable SSH.

* Boot the Pi for the first time and complete the on-screen setup. Connect to the Pi either with a monitor and keyboard, or remotely via SSH from the development machine.

* Update the system: run sudo apt update followed by sudo apt full-upgrade \-y, and reboot.

* Enable the camera interface if a Pi Camera Module is used: run sudo raspi-config, navigate to Interface Options, Camera, and enable. Reboot.

* Verify the camera works (Pi Camera: rpicam-hello; USB webcam: ls /dev/video\* and a quick capture test with v4l2-ctl or OpenCV).

## **Part B) Install the Software Stack**

* Create and activate a Python virtual environment for the project (recommended).

* Install tflite\_runtime using pip. Make sure the wheel matches the system Python version and the aarch64 architecture.

* Install the camera library: picamera2 if using a Pi Camera Module, or opencv-python if using a USB webcam.

* Install supporting libraries: numpy and Pillow.

* Confirm gpiozero is installed and importable (it ships with Raspberry Pi OS by default).

* Verify the install by importing tflite\_runtime.Interpreter and gpiozero in a Python shell.

## **Part C) Plan and Wire the Hardware**

* Count the number of output classes (N) in the Lab 9 model. This determines the number of LEDs required.

* Choose one GPIO pin for the button and N distinct GPIO pins for the LEDs. Use only the safe general-purpose pins listed in the Discussion (GPIO 4, 5, 6, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27).

* Fill in the GPIO assignment table below with the chosen pin for each role.

| Role | Component | GPIO Pin (BCM) | Physical Pin |
| :---- | :---- | :---- | :---- |
| **Capture trigger** | Push button |   |   |
| **Class 0 indicator** | LED — color: \_\_\_\_ |   |   |
| **Class 1 indicator** | LED — color: \_\_\_\_ |   |   |
| **Class 2 indicator** | LED — color: \_\_\_\_ |   |   |
| **Class 3 indicator (if applicable)** | LED — color: \_\_\_\_ |   |   |
| **Class 4 indicator (if applicable)** | LED — color: \_\_\_\_ |   |   |
| **Additional classes (if applicable)** | LED — color: \_\_\_\_ |   |   |

* Wire the button between the chosen GPIO pin and GND. The internal pull-up resistor will be enabled in software, so no external resistor is needed.

* Wire each LED: chosen GPIO pin → 330 Ω resistor → LED anode (long leg) → LED cathode (short leg) → GND.

* Double-check polarity and resistor placement before powering the Pi. A reversed LED simply will not light; a missing resistor can damage both the LED and the GPIO pin.

* Take a clear photograph of the completed breadboard and Pi setup. Save it to the report as a figure.

## **Part D) Transfer the Lab 9 Deployment Package to the Pi**

* Copy the deployment\_package/ directory from the development machine to /home/\<user\>/lab10\_deployment/ on the Raspberry Pi. Use scp, rsync, or a USB drive.

* Confirm that all five files are present: model.tflite, labels.txt, preprocessing.txt, model\_card.md, sample\_input.jpg.

* Run a sanity check: load model.tflite with the Interpreter, run one inference on sample\_input.jpg (preprocessed exactly per preprocessing.txt), and compare the predicted class to the value documented in model\_card.md. The prediction must match before proceeding.

## **Part E) Implement the Inference Loop**

Write a single Python script that performs the full capture, preprocess, inference, LED-output loop. The script should be structured around the six-step pipeline described in Section 7 of the Discussion. Specific requirements:

* At startup: load model.tflite, allocate tensors, perform one dummy (warm-up) inference on a blank input, and initialize all GPIO objects (the Button and the N LEDs).

* Main loop: wait for a button press; on press, capture a frame from the camera, preprocess it exactly as specified in preprocessing.txt, run inference, and decode the output.

* Output: drive the LEDs according to the task-specific rule chosen in Part F. Hold the LED state for a fixed display interval (3–5 seconds) before turning all LEDs off and returning to the wait state.

* Logging: print to the console each prediction (class label and confidence, or detection list, or per-class pixel counts), along with measured inference time in milliseconds.

* Graceful exit: trap Ctrl-C to turn off all LEDs and release the camera before exiting.

## **Part F) Implement the Task-Specific Output Mapping**

Implement the LED-driving logic for the task type of the model. Use the table from Section 6 of the Discussion as the specification. Document the rule and any thresholds used (e.g., confidence threshold for classification, IoU and confidence thresholds for detection NMS, pixel-fraction threshold for segmentation) directly in the script’s comments and in the report.

## **Part G) Test the End-to-End System**

Run at least six test cases on the deployed system. The set must include one input per class plus at least one challenging case (ambiguous subject, edge case, or out-of-distribution input). Record the result for each case in the table below.

| \# | Input Scenario | Expected LED Behavior | Actual LED Behavior | Pass / Fail |
| :---- | :---- | :---- | :---- | :---- |
| **1** |   |   |   |   |
| **2** |   |   |   |   |
| **3** |   |   |   |   |
| **4** |   |   |   |   |
| **5** |   |   |   |   |
| **6** |   |   |   |   |

Capture a short video (15–30 seconds) of the system in operation — showing the camera being aimed, the button being pressed, and the LEDs responding — and include it as a supplementary deliverable.

## **Part H) Benchmark On-Device Performance**

Measure on-device performance over at least 30 inferences (after warm-up). Compute mean and standard deviation. Compare against the development-machine numbers reported in Lab 9\. Fill in the table below.

| Metric | Dev Machine (from Lab 9\) | Raspberry Pi (this lab) |
| :---- | :---- | :---- |
| **Mean inference time (ms / sample)** |   |   |
| **Inference time std. dev. (ms)** |   |   |
| **End-to-end response time (button → LED, ms)** |   |   |
| **Cold-start time (first inference, ms)** |   |   |
| **Peak RAM during inference (MB)** |   |   |
| **Primary task metric (accuracy / mAP / mIoU)** |   |   |

## 

## **A. System Description**

Describe the assembled system: the Raspberry Pi model used, the camera type (Pi Camera or USB), the GPIO pin assignments from Part C, and the chosen LED color mapping. Include the photograph of the wired breadboard and the on-device sanity-check output from Part D.

## **B. End-to-End Behavior**

Discuss the test results recorded in the test-cases table. Identify which classes the system handled reliably and which were error-prone on the Pi. Cross-reference the failures against the Lab 8 evaluation — if the model already struggled with a class on the development machine, the same struggle is expected here. Distinguish genuine model errors from system-level issues such as camera focus, lighting, debounce timing, or LED-mapping bugs.

## **C. Performance and Accuracy**

Discuss the benchmark results. Compare the on-Pi inference latency to the Lab 9 dev-machine number and explain the gap (CPU architecture, no GPU, kernel optimization differences). Compare the on-Pi primary task metric against the Lab 9 number; in the absence of quantization there should be no meaningful drop, so any observed drop should be diagnosed (most often a preprocessing mismatch).

## **D. Failure Modes and Limitations**

Document at least three concrete limitations observed during deployment. Consider: classes that the LED scheme cannot disambiguate, situations where the button is missed or double-counted, lighting or camera issues that hurt accuracy, classes whose pixel counts in segmentation rarely cross the threshold, or memory/latency limits that would constrain a larger model.

# **Questions (Answer Individually)**

1. Why is tflite\_runtime preferred over the full tensorflow package on a Raspberry Pi? Address binary size, dependency footprint, and runtime performance in your answer.

2. Describe how you wired one LED and the push-button. Justify the 330 Ω current-limiting resistor for the LED and explain why no external resistor is required for the button.

3. Explain how the LED-output mapping differs across classification, object detection, and segmentation. Refer specifically to your group’s task type and the rule you implemented in Part F.

4. The first inference on the Pi is typically slower than subsequent ones. Explain why, and describe the technique you used in Part E to keep the user from experiencing this cold-start latency.

5. Compare the on-Pi inference latency to the dev-machine latency you measured in Lab 9\. What specific factors account for the difference, and what design choice in Labs 6–8 had the biggest influence on the latency observed here?

Write a conclusion of 8–10 sentences in paragraph form that summarizes:

* The hardware setup built (Pi model, camera, button, LEDs, GPIO assignment), the task type the deployed model handles, and the LED-output mapping rule chosen with any thresholds used.

* The on-device behavior observed during testing, including reliable cases and failure modes.

* The benchmark results (on-Pi latency, end-to-end response time, task-metric retention versus Lab 9), and a brief reflection on the full Lab 6–10 pipeline.

