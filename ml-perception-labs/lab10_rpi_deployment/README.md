# Lab 10 Raspberry Pi Deployment Package

This folder contains the Lab 10 retraining, evaluation, deployment, and report materials for the five-class grocery classifier:

1. Noodles
2. Rice
3. bottled water
4. canned goods
5. combo

The current copied `deployment_package/model.tflite` starts as the Lab 9 baseline. After GPU retraining, replace it with the newly converted TFLite model before copying the package to the Raspberry Pi.

## Main Files

- `notebook/Lab10_Retrain_Deploy.ipynb` - GPU retraining and deployment notebook.
- `scripts/train_lab10_gpu.py` - command-line GPU training and evaluation script.
- `scripts/inference_app.py` - Raspberry Pi button/camera/LED inference app.
- `scripts/offline_eval_tflite.py` - optional TFLite evaluator for folder-based image tests.
- `scripts/inspect_debug_capture.py` - checks Pi debug images for brightness, glare, and darkness.
- `docs/run_guide.md` - step-by-step guide for training, evaluation, conversion, and Pi deployment.
- `docs/lab10_report_draft.md` - report-ready draft aligned with `lab 10.md`.
- `docs/lab10_answers.md` - direct answers to the lab reflection questions.

## Recommended GPU Training Path

Open the notebook so you can watch training:

```powershell
jupyter notebook ml-perception-labs\lab10_rpi_deployment\notebook\Lab10_Retrain_Deploy.ipynb
```

Run cells top to bottom. The training section is named `GPU Model Training`.

## Optional GPU Training Command

```powershell
python ml-perception-labs\lab10_rpi_deployment\scripts\train_lab10_gpu.py --epochs 18 --img-size 128 --batch-size 32 --num-workers 0
```

The script requires CUDA by default and will stop instead of silently training on CPU.
