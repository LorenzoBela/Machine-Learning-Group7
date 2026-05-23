# Lab 10 Troubleshooting Notes

## Model Predicts Mostly Canned Goods

Likely causes:

- The old Lab 9 model is still deployed.
- The object is centered/cropped in a way that favors cylindrical reflective features.
- The new model was trained but not converted to TFLite.
- LED pins do not match label order.

Fix:

- Verify `model_card.md` says Lab 10 MobileNetV3-small after retraining.
- Re-run TFLite conversion.
- Re-run `offline_eval_tflite.py`.
- Check `debug_crop.jpg` to verify the object is visible.

## Prediction Is Correct Offline But Wrong on Pi

Likely causes:

- Camera color order or normalization mismatch.
- Pi image is too bright, dark, blurry, or cropped.
- Different input size between notebook and Pi script.

Fix:

- Compare `preprocessing.txt` with `scripts/inference_app.py`.
- Use `inspect_debug_capture.py` on `debug_raw.jpg` and `debug_crop.jpg`.
- Keep the object centered and fill most of the square crop.

## Confidence Is Low

Likely causes:

- Ambiguous object.
- Strong glare.
- Background clutter.
- The item looks unlike the original dataset.

Fix:

- Improve lighting and reduce glare.
- Use a plain background.
- Take multiple test captures for the report and document the limitation.

## LED Is Wrong But Console Prediction Is Right

Likely cause:

- Physical LED wiring does not match `LED_PINS`.

Fix:

- Use the label order in `labels.txt`.
- Test each GPIO pin manually.
- Update `LED_PINS` only after confirming the breadboard wiring.

