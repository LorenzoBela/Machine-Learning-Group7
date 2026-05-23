# Lab 10 MobileNetV3-small Model Card

Source model: GPU retrained PyTorch MobileNetV3-small

Dataset: Lab04 EDA Bias Dataset copied from Lab 8; no new Pi-camera training images were collected.

Task type: 5-class image classification

Classes: Noodles, Rice, bottled water, canned goods, combo

Input shape: RGB 128x128, normalized with ImageNet mean/std

Output format: logits over 5 classes in labels.txt order

Selection rule: best validation macro F1 plus worst-class recall score

Test accuracy: 0.9600

Macro F1: 0.9660

Deployment note: run TFLite conversion below before replacing model.tflite on the Raspberry Pi.
