# Lab 10 Deployment Checklist

## Before Training

- Confirm CUDA is available on the laptop.
- Confirm dataset exists at `ml-perception-labs/lab08_finetuning/data/raw`.
- Confirm class folders are exactly `Noodles`, `Rice`, `bottled water`, `canned goods`, `combo`.

## After Training

- Check `outputs/logs/lab10_training_summary.json`.
- Check `outputs/tables/lab10_test_classification_report.csv`.
- Confirm accuracy improves over 72%.
- Confirm weak classes are documented if any recall is below 75%.

## Before Pi Testing

- Confirm `deployment_package/model.tflite` is the converted Lab 10 model, not the old Lab 9 baseline.
- Confirm `deployment_package/labels.txt` order is:
  - `Noodles`
  - `Rice`
  - `bottled water`
  - `canned goods`
  - `combo`
- Confirm `preprocessing.txt` input size matches the model.
- Confirm `LED_PINS = [4, 5, 6, 12, 13]` matches the physical LED order.

## Pi Hardware

- Button on GPIO 17 to GND.
- Class 0 LED on GPIO 4.
- Class 1 LED on GPIO 5.
- Class 2 LED on GPIO 6.
- Class 3 LED on GPIO 12.
- Class 4 LED on GPIO 13.
- Each LED uses a 330 ohm resistor.

## Final Evidence

- Photograph of Pi and breadboard.
- Screenshot or terminal log of sanity check.
- Six test-case table.
- 15-30 second demo video.
- Latency and end-to-end response table.

