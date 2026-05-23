# MobileNetV2 m1_t3 Model Card

Source model: Lab 8 fine-tuned PyTorch MobileNetV2 trial m1_t3

Dataset: Lab04 EDA Bias Dataset (bottled water, canned goods, combo, Noodles, Rice)

Task type: 5-class image classification

Classes: bottled water, canned goods, combo, Noodles, Rice

Input shape: original PyTorch `(1, 3, 64, 64)` normalized RGB; converted TFLite input `[1, 64, 64, 3]`

Output format: logits over 5 classes; converted TFLite output `[1, 5]`

Expected test metric: Lab 8 original test accuracy 72.00%; converted TFLite test accuracy 72.00%

Conversion pathway: PyTorch checkpoint -> ONNX -> TensorFlow SavedModel -> TensorFlow Lite float32
