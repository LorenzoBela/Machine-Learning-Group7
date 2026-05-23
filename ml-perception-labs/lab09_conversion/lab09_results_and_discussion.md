# Lab 09 Conversion: Results, Discussion, and Conclusion

## 1. Results Summary

Lab 9 carried over the Lab 8 deployment recommendation: **MobileNetV2 trial m1_t3** trained on the same 5-class grocery dataset. The dataset remained at 100 images per class, and the same seed-42 60/20/20 split produced 300 training samples, 100 validation samples, and 100 held-out test samples. The source checkpoint was copied from `lab08_finetuning/finetuned_models/MobileNetV2_m1_t3.pth` into the Lab 9 `source_model/` folder before conversion.

| Metric | Original Model (Lab 8) | Converted TFLite Model |
| :--- | :--- | :--- |
| Model file size (MB) | 8.75 | 8.47 |
| Primary task metric (accuracy) | 72.00% | 72.00% |
| Macro F1 | 72.69% | 72.69% |
| Max absolute output error vs. original | 0.000000 | 0.000014 |
| Prediction agreement rate (%) | 100.00% original reference | 100.00% |
| Mean inference latency (ms / sample) | 5.4222 | 1.3041 |

The converted TFLite input shape reported by the interpreter was `[1, 64, 64, 3]`, and the output shape was `[1, 5]`. The side-by-side prediction figure was saved to `C:/Users/Lorenzo Bela/Downloads/Elective Machine Learning/ml-perception-labs/lab09_conversion/outputs/figures/lab09_original_vs_tflite_examples.png`.

---

## 2. Conversion Pathway Summary

The source framework was **PyTorch**, so the conversion used the required cross-framework route: PyTorch checkpoint to ONNX, ONNX to TensorFlow SavedModel, and TensorFlow SavedModel to TensorFlow Lite. The PyTorch export used a fixed dummy input of `(1, 3, 64, 64)`, ONNX opset 17, input name `input`, and output name `logits`. The ONNX model was converted to TensorFlow SavedModel using `onnx2tf`, then converted to a float32 `.tflite` file using `tf.lite.TFLiteConverter.from_saved_model` with no optimization or quantization settings.

No quantization was applied, so the output values are expected to remain close to the original PyTorch logits. The converter log was saved as `onnx2tf_conversion.log`. FlexOp note: No FlexOps were detected in the converter log.

---

## 3. Parity and Accuracy

The original PyTorch model reached **72.00%** accuracy and **72.69%** macro F1 on the Lab 8 held-out test split. The converted TFLite model reached **72.00%** accuracy and **72.69%** macro F1 on the same test samples. The absolute accuracy drop was **0.00%**, which is a relative drop of **0.00%** from the original model.

The maximum absolute output error across the evaluated test logits was **0.000014**, and the prediction agreement rate was **100.00%**. These values show that the TFLite model is a faithful replacement for the original model on this test split.

---

## 4. Failure Modes and Limitations

The main conversion risk was input layout. The PyTorch model was exported with NCHW input, while TensorFlow Lite commonly reports NHWC input after conversion. The verification code handled this by inspecting the TFLite interpreter input shape and feeding normalized test images in the layout expected by the converted model.

Two limitations remain for Lab 10. First, the dataset is still small at 500 total images, so the 72% test accuracy should not be treated as production-grade robustness. Second, this conversion was verified on the development machine, not on the Raspberry Pi; Lab 10 should re-check latency, memory use, and preprocessing consistency on the actual Pi runtime.

---

## 5. Questions

**Why is it necessary to convert the Lab 8 model to TensorFlow Lite for Raspberry Pi deployment instead of running the original framework directly on the Pi?**

The Lab 8 model was trained and saved in PyTorch, which is heavier than needed for inference on a Raspberry Pi. TensorFlow Lite packages the model into a compact inference-only file and can run through a smaller runtime on ARM hardware. In this lab, the PyTorch checkpoint was 8.75 MB and the TFLite file was 8.47 MB, but the practical benefit is the simpler deployment dependency and edge-oriented interpreter rather than file size alone.

**Explain why a TensorFlow SavedModel is produced as an explicit intermediate even when some tools can convert from PyTorch or Keras directly to TFLite.**

The TensorFlow Lite converter is designed to consume TensorFlow graphs reliably, so the SavedModel acts as the canonical TensorFlow representation before final TFLite conversion. For this PyTorch source model, ONNX served as the bridge format and `onnx2tf` produced the SavedModel. Keeping the SavedModel as an explicit artifact also gives a recoverable checkpoint if the final TFLite conversion or Pi-side testing needs to be repeated.

**Based on your parity-test results, is the converted `.tflite` a faithful replacement for the original model? Justify your answer using the maximum absolute error and the task metric reported in Part E.**

The converted model is faithful on the measured Lab 8 test split. The maximum absolute output error was **0.000014**, the prediction agreement rate was **100.00%**, and the TFLite accuracy was **72.00%** compared with **72.00%** for the original model. Those measured values are the basis for accepting the converted file for Lab 10 handoff.

## 6. Conclusion

Lab 9 converted the Lab 8 recommended **MobileNetV2 trial m1_t3** model from PyTorch to TensorFlow Lite using the ONNX and TensorFlow SavedModel intermediate formats. The same Lab 8 dataset, class order, seed-42 split, 64x64 RGB resizing, and ImageNet normalization were used so that the verification matched the original experiment. The final conversion produced `saved_model/` and `tflite_model/model.tflite`, with no quantization or optimization flags applied. The parity test produced a maximum absolute output error of **0.000014** and a prediction agreement rate of **100.00%**. The original model reached **72.00%** test accuracy, while the converted TFLite model reached **72.00%**, giving an absolute drop of **0.00%**. The deployment package contains `model.tflite`, `labels.txt`, `preprocessing.txt`, `model_card.md`, and `sample_input.jpg`. The main open risks for Lab 10 are Raspberry Pi runtime latency, memory use, and exact preprocessing consistency on the target device. The converted TFLite model is ready for Pi-side sanity testing using the packaged sample input.
