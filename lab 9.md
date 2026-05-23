MACHINE LEARNING AND PERCEPTION LAB**

Adamson University Computer Engineering Department

**Laboratory Exercise 9**

**Conversion of the Fine-Tuned Model to TensorFlow and TensorFlow Lite Format**

Submitted by:

**Group #**

| **Category**                                               | **Exceptional**<br><br>**4**                                                                                                                                                    | **Acceptable**<br><br>**3**                                                                   | **Marginal**<br><br>**2**                                                                                    | **Unacceptable**<br><br>**1**                                                   | **Score** |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- | --------- |
| **System / Pipeline Design & Implementation (30%)**        | Clear, well-structured machine learning pipeline or experimental design that fully meets the stated objectives, requirements, and constraints of the lab.                       | Adequate pipeline or experimental design with minor limitations; meets most lab requirements. | Partial or loosely structured design; some requirements addressed but key elements are missing or incorrect. | Minimal or unclear design effort; does not address the lab requirements.        |           |
| **Application of Tools & Techniques**<br><br>**(25%)**     | Correct selection and expert use of appropriate tools and techniques (e.g., Python, Jupyter, ML libraries, data analysis tools); methods are effectively applied and justified. | Correct tool selection with minor errors or inconsistencies in application.                   | Limited, inappropriate, or incorrect tool usage; techniques partially support the task.                      | No meaningful or incorrect use of required tools and techniques.                |           |
| **Implementation & Resource Utilization**<br><br>**(20%)** | Efficient, logical, and well-organized implementation; methods and resources are fully aligned with the problem and constraints.                                                | Functional implementation with minor inefficiencies or redundancies.                          | Implementation partially works but lacks efficiency, clarity, or completeness.                               | Poor or non-functional implementation with little consideration of constraints. |           |
| **Testing, Analysis & Validation**<br><br>**(15%)**        | Comprehensive testing and analysis; results are clearly validated, interpreted, and supported by appropriate metrics, figures, or tables.                                       | Adequate testing and analysis with mostly correct interpretation of results.                  | Limited testing; analysis is incomplete, weakly supported, or partially incorrect.                           | No testing performed or results are incorrectly analyzed or interpreted.        |           |
| **Documentation & Reporting**<br><br>**(10%)**             | Clear, complete, and well-structured lab report/notebook with proper figures, tables, explanations, and reflection.                                                             | Complete documentation with minor issues in clarity, organization, or detail.                 | Partial documentation; missing sections, unclear explanations, or poor organization.                         | Incomplete, poorly written, or missing documentation.                           |           |
| **TOTAL SCORE**                                            |                                                                                                                                                                                 |                                                                                               |                                                                                                              |                                                                                 |           |

| **Group Members** | | | |
| --- | | | | --- | --- | --- |
| **STUDENT NUMBER** | **NAME** | **CONTRIBUTION** | **SCORE** |
| | | | |
| | | | |
| | | | |
| | | | |

Submitted to:

Engr. Dexter James L. Cuaresma

Date:

mm/dd/year

**OBJECTIVES**

- Convert the final fine-tuned model from Laboratory Exercise 8 into the TensorFlow SavedModel format.
- Convert the TensorFlow SavedModel into the TensorFlow Lite (.tflite) format using the standard TFLite Converter.
- Verify that the converted model produces the same outputs and retains the same task metric as the original.

**DISCUSSION**

# **Introduction**

In Laboratory Exercise 8, the group fine-tuned a deployment-ready model on a development machine using PyTorch, Keras, scikit-learn, or Ultralytics. To run that model on a Raspberry Pi in Laboratory Exercise 10, it must be converted into a lightweight format that does not require the full training framework as a dependency.

TensorFlow Lite (TFLite) is the standard format for machine learning inference on the Raspberry Pi. It produces a single small file (.tflite) and runs on a lightweight library (tflite_runtime) that installs cleanly on Raspberry Pi OS. This lab converts the Lab 8 model into the TFLite format. No quantization or other compression is applied - the goal is a faithful, easy-to-verify conversion that Lab 10 will deploy directly.

# **Detailed Discussion**

## **1\. Why TensorFlow Lite for the Raspberry Pi**

Research frameworks such as PyTorch and TensorFlow are designed for flexibility during training: they support dynamic computation graphs, automatic differentiation, large parallel hardware, and rapid experimentation. These properties come at a cost - large binary size, heavy dependency footprint, and runtime overhead - that is acceptable on a development workstation but problematic on a Raspberry Pi, where memory, storage, and CPU are all constrained.

TensorFlow Lite (TFLite) was designed to solve exactly this problem. It targets inference only, runs on a small native library (the tflite_runtime package is roughly 1-5 MB depending on platform, compared to several hundred megabytes for the full tensorflow package), uses a static graph representation, and ships with kernels that are heavily optimized for ARM CPUs and other embedded hardware.

The same .tflite file can also run on Android, iOS, embedded Linux (including Raspberry Pi OS), and microcontrollers, which makes TFLite a portable target across the entire edge-deployment spectrum. Choosing TFLite for this lab is therefore both pragmatic - it works well on the Pi - and pedagogical, because the workflow students learn here transfers to many other edge platforms.

## **2\. The TensorFlow Lite Ecosystem**

Three components make up the TFLite ecosystem, and each plays a distinct role in the workflow of this lab.

The TFLite Converter is the offline tool that runs on the development machine. Its Python API (tf.lite.TFLiteConverter) accepts a TensorFlow SavedModel, a Keras model, or a set of concrete functions, and produces a single .tflite file. The converter is also where graph optimizations such as operator fusion and constant folding are applied automatically.

The TFLite Runtime is the inference-only library installed on the target device. On the Raspberry Pi, the tflite_runtime package is installed instead of the full tensorflow package. This is a critical distinction, because tensorflow itself has dependencies that are large or unavailable on ARM Linux. The runtime exposes an Interpreter class that loads a .tflite file, allocates the necessary tensors, and runs inference.

The TFLite Delegates are pluggable acceleration backends. On a Raspberry Pi, the relevant delegate is XNNPACK - a highly optimized library of CPU kernels for ARM that is enabled by default and provides substantial speed-ups for standard float-32 inference. Other delegates exist (GPU, NNAPI, Edge TPU), but they apply to other hardware targets and are not used in this lab.

## **3\. The TensorFlow SavedModel as the Canonical Intermediate**

Every conversion in this lab passes through a TensorFlow SavedModel directory before producing the .tflite file. The SavedModel is TensorFlow's canonical, language-neutral representation of a model. Unlike a single-file format such as Keras's .h5, a SavedModel is a directory containing several files: saved_model.pb (the serialized computation graph), variables/ (the weights), assets/ (auxiliary files), and signatures (named input/output specifications).

Keeping the SavedModel as an explicit intermediate provides three benefits. First, it is inspectable: it can be loaded back into TensorFlow and re-evaluated, opened in Netron, or queried with the saved_model_cli command-line tool. Second, it is reproducible: it captures the entire model in a portable format independent of the original training framework. Third, it is recoverable: if the final .tflite conversion fails on a specific operator, the SavedModel remains a known-good artifact that students can return to without re-running the upstream steps.

This is why even tools that advertise direct PyTorch-to-TFLite conversion still produce a SavedModel internally. The SavedModel is the only input format the TFLite Converter is designed to consume reliably across model architectures.

## **4\. ONNX: The Bridge for Cross-Framework Conversion**

Models trained in PyTorch, scikit-learn, or other non-TensorFlow frameworks cannot be opened by the TFLite Converter directly. To bridge the gap, the model is first exported to ONNX (Open Neural Network Exchange) - a portable, framework-neutral graph format jointly maintained by major machine learning organizations.

ONNX defines a standard operator set: each operator (Conv2D, MatMul, Relu, Softmax, and so on) has a specified mathematical definition and a versioned schema. When a PyTorch model is exported with torch.onnx.export, every layer is decomposed into ONNX operators and packed into a single .onnx file. Tools such as onnx2tf then read this file and emit an equivalent TensorFlow SavedModel.

Two consequences are worth understanding. First, ONNX has very wide coverage of standard deep learning operators, so most common architectures convert cleanly. Custom layers, framework-specific operators, or recent research operators may not have ONNX equivalents, and these are the most common cause of export failure. Second, ONNX is also widely supported as a deployment format in its own right - ONNX Runtime is a competing edge runtime - but this lab does not deploy ONNX directly because TFLite has better ARM-CPU support and a smaller deployment footprint on Raspberry Pi OS.

## **5\. The Conversion Pipeline**

Every conversion in this lab follows the same general shape:

- Source model (PyTorch / Keras / scikit-learn / Ultralytics)
- Intermediate format (ONNX) - only required when the source framework is not TensorFlow
- TensorFlow SavedModel - the canonical TensorFlow representation, produced as an explicit intermediate
- TensorFlow Lite (.tflite) - the final deployment file

The exact first step depends on the source framework. The table below maps each likely source from Lab 8 to its conversion path.

| **Source Framework**   | **Conversion Path**                  | **Primary Tools**                                   |
| ---------------------- | ------------------------------------ | --------------------------------------------------- |
| **TensorFlow / Keras** | Keras model → SavedModel → TFLite    | tf.saved_model, tf.lite.TFLiteConverter             |
| **PyTorch**            | PyTorch → ONNX → SavedModel → TFLite | torch.onnx.export, onnx2tf, tf.lite.TFLiteConverter |
| **Ultralytics YOLO**   | Framework-native export to TFLite    | ultralytics (model.export with format='tflite')     |
| **scikit-learn**       | sklearn → ONNX → SavedModel → TFLite | skl2onnx, onnx2tf, tf.lite.TFLiteConverter          |

## **6\. What the TFLite Converter Does**

It is tempting to think of the TFLite Converter as a simple "save in a different format" step, but it actually performs several optimization passes on the graph as part of conversion. Understanding these passes helps explain why the resulting .tflite file may be smaller and structurally different from the SavedModel even when no quantization is applied.

- Constant folding: operations whose inputs are all constants are evaluated at conversion time and replaced with their result. A chain of reshape operations on a fixed weight tensor is collapsed into a single pre-shaped constant.
- Operator fusion: common sequences of operators are combined into single fused operators. The most well-known example is Conv2D → BatchNorm → ReLU, which fuses into a single FusedConv2D operator. Fusion reduces both the number of operator dispatches at runtime and the number of intermediate tensors that must be allocated.
- Dead code elimination: operators whose outputs are not used by any downstream operator are removed.
- Operator lowering: high-level TensorFlow operators are translated into the TFLite Builtin operator set, which is smaller and more uniform.

If a TensorFlow operator does not have a TFLite Builtin equivalent, the converter can either fail or insert a FlexOp (also called a "TF Select" operator). FlexOps execute by calling back into the full TensorFlow runtime, which increases the runtime size required on the Pi. The presence of FlexOps in the converted .tflite is acceptable but should be flagged in the report because it changes the deployment dependency requirements for Lab 10.

## **7\. Verifying the Conversion**

A model that converts without error is not necessarily a correct model. Operator fusion, graph simplification, and small numerical differences between the source framework, the SavedModel runtime, and the TFLite Interpreter can all introduce subtle drift. Two checks are required before the .tflite file is considered deployment-ready:

- Numerical parity - run the same input through both the original and converted models, and confirm that the outputs match within a small numerical tolerance (close to machine epsilon for float32). Since no quantization is applied in this lab, the expected drift should be very small.
- Metric retention - re-evaluate the converted model on the Lab 8 test set using the primary task metric (accuracy, mAP, or mIoU) and confirm that the metric did not drop meaningfully.

If parity fails, the cause is almost never the conversion itself - it is most often a mismatch in the preprocessing pipeline: a different resize, different normalization values, or a different channel order (RGB vs. BGR) between the training and inference scripts. The preprocessing recipe must be carried over verbatim from Lab 8.

## **8\. Common Pitfalls**

- Mismatched preprocessing is the single most common cause of unexpected accuracy loss after a clean conversion. Resize order, interpolation method, normalization mean and standard deviation, and channel order (RGB vs. BGR) must all match the training pipeline exactly.
- Unsupported or custom operators may force the converter to insert FlexOps or to fail outright. Inspecting the converter log carefully and replacing custom operations with standard equivalents is the usual fix.
- Dynamic input shapes are supported by TFLite but complicate downstream tooling, including Netron inspection and Pi-side preprocessing. Exporting with a fixed input shape is strongly preferred unless dynamic shape is essential.
- BatchNorm folding: BatchNorm layers are normally folded into the preceding convolution during conversion. If a model uses non-standard BatchNorm placements, folding may not occur, leaving extra runtime overhead. Inspecting the .tflite graph in Netron confirms whether folding succeeded.
- Skipping parity checks: a model that converts without error is not the same as a model that behaves correctly. Numerical parity testing is mandatory before declaring the conversion complete.

**MATERIALS**

## **Hardware**

- Laptop/PC with at least 8 GB RAM (GPU recommended for deep learning models)
- Google Colab (free tier or Colab Pro) is acceptable for GPU-accelerated training

## **Software**

- Python 3.10+
- Jupyter Notebook / Google Colab

## **Libraries**

- tensorflow - provides tf.lite.TFLiteConverter
- onnx and onnx2tf - used when the source is PyTorch or scikit-learn
- ultralytics - used when the source is a YOLO model
- skl2onnx - used when the source is a scikit-learn model
- numpy, pillow, matplotlib

## **Dataset / Data Source**

- The final fine-tuned model weights produced in Laboratory Exercise 8
- The same dataset and the same test split used in Laboratory Exercises 6-8

**PROCEDURES**

## **Part A) Project Setup**

- Create a directory: ml-perception-labs/lab09_conversion/
- Replicate the following folder structure:

lab09_conversion/

├── source_model/ fine-tuned weights from Lab 8

├── saved_model/ TensorFlow SavedModel (produced in Part C)

├── tflite_model/ model.tflite (produced in Part D)

├── deployment_package/ files handed off to Lab 10

└── notebook/

└── Lab09_Conversion.ipynb

- Create the notebook Lab09_Conversion.ipynb. The first cell must display: Name, Section, Date, Dataset, and the model carried over from Lab 8.

## **Part B) Identify the Source Model and Conversion Pathway**

Carry over the single model recommended for deployment in the Lab 8 conclusion. Fill in the table below before starting any conversion.

| **Item**                                                  | **Value** |
| --------------------------------------------------------- | --------- |
| **Model name (from Lab 8)**                               |           |
| **Source framework**                                      |           |
| **Task type (classification / detection / segmentation)** |           |
| **Input shape (channels, height, width)**                 |           |
| **Preprocessing (resize, normalization, color order)**    |           |
| **Number of classes / categories**                        |           |
| **Baseline test metric (from Lab 8)**                     |           |
| **Chosen conversion pathway**                             |           |

## **Part C) Convert to TensorFlow SavedModel**

Produce a TensorFlow SavedModel directory at saved_model/. The exact tool depends on the source framework chosen in Part B.

- For PyTorch and scikit-learn sources: first export the model to ONNX with a fixed input shape, then convert the ONNX file to a SavedModel using onnx2tf.
- For Keras / TensorFlow sources: save directly using tf.saved_model.save.
- For Ultralytics YOLO: use the framework-native export, which produces the SavedModel as part of the .tflite export pipeline.
- Open the SavedModel in Netron and confirm that the input shape, output shape, and operators match the source model.
- Run one forward pass through the SavedModel and confirm that its output matches the source model on the same input.

## **Part D) Convert to TensorFlow Lite**

Produce the .tflite file using the standard TFLite Converter. No optimization flags are enabled - the converter should produce a faithful float32 conversion of the SavedModel.

- Configure tf.lite.TFLiteConverter.from_saved_model with no optimization or quantization settings.
- Set the input and output data types to float32.
- Run the conversion and save the file as tflite_model/model.tflite.
- Open the .tflite file in Netron and confirm the graph structure looks correct. Record any FlexOps (TF Select) reported in the converter log.

## **Part E) Verify the Converted Model**

Confirm that the converted .tflite behaves like the original model. Both checks below are required.

- Numerical parity: run a batch of 50-100 test images through both the original model and the .tflite model. Record the maximum absolute error between their outputs. The error should be very small (close to machine epsilon).
- Metric retention: re-evaluate the .tflite model on the same held-out test set used in Lab 8, using the same primary metric (accuracy, mAP, or mIoU). Record the absolute and relative drop versus the original.
- Save at least four side-by-side example predictions (original vs. .tflite) to the notebook.
- Fill in the comparison table below and save it as deployment_package/conversion_comparison.csv.

| **Metric**                                      | **Original Model (Lab 8)** | **Converted TFLite Model** |
| ----------------------------------------------- | -------------------------- | -------------------------- |
| **Model file size (MB)**                        |                            |                            |
| **Primary task metric (accuracy / mAP / mIoU)** |                            |                            |
| **Max absolute output error vs. original**      |                            |                            |
| **Prediction agreement rate (%)**               |                            |                            |
| **Mean inference latency (ms / sample)**        |                            |                            |

## **Part F) Package for Lab 10**

Assemble the deployment package in the deployment_package/ directory. The package must contain:

- model.tflite - the converted model file.
- labels.txt - class names, one per line, in the order matching the model output.
- preprocessing.txt - the exact preprocessing recipe (input size, normalization mean/std, channel order).
- model_card.md - a short description of the model: source, task type, input shape, output format, expected test metric, and the dataset name.
- sample_input.jpg - one representative test image for sanity checking on the Pi.

**RESULTS AND DISCUSSION**

## **A. Conversion Pathway Summary**

State the source framework, the steps performed in Parts C and D, and the tools used at each step. Include screenshots of the SavedModel and the .tflite file viewed in Netron. Note any operators that required special handling.

## **B. Parity and Accuracy**

Report the maximum absolute output error and the prediction agreement rate. Compare the primary task metric between the original and the converted model using the table from Part E, and discuss whether the difference is acceptable for the task.

## **C. Failure Modes and Limitations**

Document any failure encountered during conversion (unsupported operators, dynamic shape issues, accuracy mismatches) and the workaround applied. List at least two limitations that remain in the converted model and that should be re-verified on the actual Raspberry Pi hardware in Lab 10.

# **Questions (Answer Individually)**

- Why is it necessary to convert the Lab 8 model to TensorFlow Lite for Raspberry Pi deployment instead of running the original framework directly on the Pi?
- Explain why a TensorFlow SavedModel is produced as an explicit intermediate even when some tools can convert from PyTorch or Keras directly to TFLite.
- Based on your parity-test results, is the converted .tflite a faithful replacement for the original model? Justify your answer using the maximum absolute error and the task metric reported in Part E.

Write a conclusion of 6-8 sentences in paragraph form that summarizes:

**CONCLUSION**

- The model carried over from Lab 8, the conversion pathway followed, and any operator-level issues encountered along the way.
- The parity-test results (maximum absolute error and prediction agreement rate) and the task-metric retention on the held-out test set.
- The contents of the final deployment package handed off to Lab 10, and any open risks to re-verify on the actual Raspberry Pi hardware.