# Lab 10 Step-by-Step Run Guide

## 1. Verify GPU Training Environment

Run from the workspace root:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

Expected for this machine:

```text
True
NVIDIA GeForce RTX 3050 Ti Laptop GPU
```

Do not train on CPU for the final run. The training script requires CUDA unless `--allow-cpu` is explicitly passed for debugging.

## 2. Installed Packages

The current environment has been prepared with the needed training, notebook, and conversion packages:

```powershell
python -m pip install notebook ipykernel nbformat pandas matplotlib scikit-learn opencv-python pillow onnx onnx2tf onnx_graphsurgeon sng4onnx onnxsim ai-edge-litert "tensorflow==2.20.*" "tf-keras==2.20.*" "protobuf==5.29.5"
python -m ipykernel install --user --name lab10-ml --display-name "Python (Lab10 ML)"
```

When opening the notebook, choose the kernel named:

```text
Python (Lab10 ML)
```

## 3. Train the Higher-Accuracy Model in the Notebook

Use this path if you want to see every training cell, epoch log, table, and confusion matrix.

Start Jupyter:

```powershell
python -m notebook ml-perception-labs\lab10_rpi_deployment\notebook\Lab10_Retrain_Deploy.ipynb
```

Then run the notebook top to bottom.

The important training section is titled:

```text
GPU Model Training
```

That section prints one row per epoch with:

- training loss
- training accuracy
- training macro F1
- validation loss
- validation accuracy
- validation macro F1
- worst-class recall
- checkpoint selection score

The notebook is configured to use CUDA and will raise an error if CUDA is not available.

The notebook now defaults to:

```text
SPLIT_STRATEGY = "blocked_by_filename"
```

That is stricter than the earlier Lab 8 random split because it holds out later-numbered images per class as validation/test batches. Use `lab8_random` only when you intentionally want to reproduce the earlier 96% result.

## 4. Alternative: Train from PowerShell

Run:

```powershell
python ml-perception-labs\lab10_rpi_deployment\scripts\train_lab10_gpu.py --epochs 18 --img-size 128 --batch-size 32 --num-workers 0
```

What this does:

- Loads the Lab 8 dataset from `ml-perception-labs/lab08_finetuning/data/raw`.
- Uses the same five classes and split seed as Lab 8.
- Trains MobileNetV3-small on CUDA with mixed precision.
- Uses realistic augmentation for lighting, glare, crop, perspective, blur, and object position.
- Uses weighted sampling so weak classes are not ignored.
- Selects the best checkpoint by validation macro F1 plus worst-class recall.

Outputs:

- `deployment_package/lab10_mobilenetv3_small_best.pth`
- `outputs/tables/lab10_training_history.csv`
- `outputs/tables/lab10_test_classification_report.csv`
- `outputs/tables/lab10_test_confusion_matrix.csv`
- `outputs/figures/lab10_test_confusion_matrix.png`
- `outputs/logs/lab10_training_summary.json`
- updated `deployment_package/labels.txt`
- updated `deployment_package/preprocessing.txt`
- updated `deployment_package/model_card.md`

## 5. Check Whether the New Model Beat the Baseline

Open:

```text
ml-perception-labs/lab10_rpi_deployment/outputs/logs/lab10_training_summary.json
```

Compare against the Lab 9 baseline:

```text
Baseline accuracy: 72.00%
Baseline macro F1: 72.69%
```

Target:

```text
New accuracy: ideally 85%+
Worst-class recall: ideally 75%+
```

If the result is not better, rerun with:

```powershell
python ml-perception-labs\lab10_rpi_deployment\scripts\train_lab10_gpu.py --epochs 25 --img-size 128 --batch-size 24
```

If GPU memory is tight, use:

```powershell
python ml-perception-labs\lab10_rpi_deployment\scripts\train_lab10_gpu.py --epochs 18 --img-size 96 --batch-size 32
```

## 6. Convert the Trained Model to TFLite

Open and run the final conversion cell in:

```text
ml-perception-labs/lab10_rpi_deployment/notebook/Lab10_Retrain_Deploy.ipynb
```

Expected output:

```text
deployment_package/model.tflite
```

You can also convert outside the notebook:

```powershell
.venv\Scripts\python.exe ml-perception-labs\lab10_rpi_deployment\scripts\convert_lab10_to_tflite.py
```

Important: the copied `model.tflite` is still the old Lab 9 baseline until this conversion step succeeds.

## 7. Offline TFLite Evaluation

After `model.tflite` is replaced by the converted Lab 10 model, run:

```powershell
python ml-perception-labs\lab10_rpi_deployment\scripts\offline_eval_tflite.py --data-dir ml-perception-labs\lab08_finetuning\data\raw
```

This writes:

```text
outputs/tables/tflite_eval_predictions.csv
```

Use this to verify that the converted TFLite model still performs like the PyTorch checkpoint.

Do not report the full-dataset TFLite accuracy as the final ML accuracy. Use it as a conversion sanity check. The honest model-quality number is the notebook test report from the held-out split.

## 8. Copy to Raspberry Pi

Copy these to the Pi:

```text
deployment_package/
scripts/inference_app.py
```

Recommended Pi location:

```text
/home/jhosil/lab10_deployment/
```

Example from Windows PowerShell:

```powershell
scp -r ml-perception-labs\lab10_rpi_deployment\deployment_package jhosil@<PI_IP>:/home/jhosil/lab10_deployment/
scp ml-perception-labs\lab10_rpi_deployment\scripts\inference_app.py jhosil@<PI_IP>:/home/jhosil/lab10_deployment/
```

## 9. Run on Raspberry Pi

SSH into the Pi:

```bash
ssh jhosil@<PI_IP>
cd /home/jhosil/lab10_deployment
python inference_app.py
```

Press the button once per test object. The script prints:

- predicted class
- confidence
- inference latency
- end-to-end latency
- brightness/glare warning

It also saves:

```text
/home/jhosil/lab10_deployment/debug_raw.jpg
/home/jhosil/lab10_deployment/debug_crop.jpg
```

## 10. Diagnose Bad Predictions

Copy the debug images back to the laptop and run:

```powershell
python ml-perception-labs\lab10_rpi_deployment\scripts\inspect_debug_capture.py debug_raw.jpg debug_crop.jpg
```

Common fixes:

- High glare: angle object or camera away from overhead light.
- Very dark image: add light or adjust camera exposure.
- Object cropped out: move object to center before pressing the button.
- Correct prediction but wrong LED: fix LED pin order.

## 11. Fill the Report

Use:

```text
docs/lab10_report_draft.md
docs/lab10_answers.md
```

Add your real Pi measurements:

- Pi model
- camera type
- GPIO pin colors
- inference latency
- end-to-end response time
- six test cases
- final video evidence
