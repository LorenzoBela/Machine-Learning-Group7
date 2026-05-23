# Step-by-Step Guide to Finish Lab 10

This guide outlines exactly what your group needs to do to finish Laboratory 10, using the `inference_app.py` script we just created. The lab is divided into physical execution, benchmarking, and report writing.

## 1. Physical Hardware Setup (Parts A & C)
1. **Flash OS**: Ensure your Raspberry Pi has a 64-bit Raspberry Pi OS installed.
2. **Connect Camera**: Plug your USB webcam into the Raspberry Pi.
3. **Wire the Button**: Connect a push-button to **GPIO 17** and **GND** on your breadboard.
4. **Wire the 5 LEDs**: For your 5 classes (`Noodles`, `Rice`, `bottled water`, `canned goods`, `combo`), wire 5 LEDs to **GPIO 4, 5, 6, 12, and 13**. Ensure each LED has a 330 Ω resistor connected in series to GND.
5. **Take a Photo**: Take a clear photograph of your completed breadboard and Pi setup. You will need this for **Part A** of your lab report.

## 2. Software & File Transfer (Parts B & D)
1. **Transfer Files**: Copy the `model.tflite` and `labels.txt` from your Lab 9 GitHub repository, along with the `inference_app.py` we wrote, to a folder on your Pi: `/home/pi/lab10_deployment/`.
2. **Install Dependencies**: Open a terminal on the Pi and run:
   ```bash
   sudo apt update
   pip install tflite-runtime numpy Pillow opencv-python gpiozero
   ```
3. **Sanity Check**: Run a quick test to ensure the script can load the model without crashing.

## 3. Execution & Testing (Part G)
1. **Run the Script**: `python inference_app.py`
2. **Conduct the 6 Test Cases**: Point the camera at test subjects and press the button. You need 1 test per class, plus 1 challenging/edge case.
3. **Record Results**: Note the expected vs. actual LED behavior for each of the 6 tests to fill out the table in **Part G**.
4. **Record a Video**: Capture a short 15–30 second video of someone pressing the button and the LEDs lighting up. This is a required supplementary deliverable.

## 4. Benchmarking (Part H)
1. **Capture Latency**: Keep the script running and press the button at least 30 times to get 30 inferences. Note the "Latency" printed in the terminal for each.
2. **Calculate Stats**: Calculate the mean and standard deviation of those 30 latency values.
3. **Fill the Benchmark Table**: Compare these new Raspberry Pi latency values to the latency values you measured on your PC in Lab 9.

## 5. Write the Lab Report Document
Open your Lab 10 Word/Markdown document and fill out the sections:

### Main Sections
- **A. System Description**: Describe the Pi 4/5, USB camera, and the GPIO pin assignments we used. Insert your breadboard photo here.
- **B. End-to-End Behavior**: Discuss how well the 6 test cases worked. Did it struggle with certain classes?
- **C. Performance and Accuracy**: Discuss why the Pi is slower than your PC (e.g., no GPU, ARM CPU).
- **D. Failure Modes**: Document 3 limitations (e.g., bad lighting, button debouncing, USB camera focus issues).

### Individual Questions
Each group member must answer these 5 questions individually:
1. Why `tflite_runtime` is used instead of the full `tensorflow` package (mention binary size and ARM constraints).
2. Justify the 330 Ω resistor for LEDs and why the button doesn't need an external resistor (it uses internal pull-up).
3. Explain the Classification LED mapping (we used `argmax` to find the highest probability and lit the corresponding single LED).
4. Why the first inference is slow (tensor allocation/loading), and how we fixed it (the warm-up inference at the start of our script).
5. Compare the Pi latency to your PC latency from Lab 9.

### Conclusion
Write an 8-10 sentence conclusion summarizing the hardware setup, the test behaviors, the benchmark results, and a reflection on the entire Lab 6–10 pipeline.

---
**Submission Checklist:**
- [ ] Completed Lab 10 Report (with tables, photo, answers, and conclusion)
- [ ] 15-30 second demonstration video
- [ ] Your `inference_app.py` code (usually appended to the report or submitted alongside)
