# Lab 10 YOLOv8n Object Detection Model Card

Source model: YOLOv8-nano fine-tuned on grocery dataset

Dataset: Roboflow ML project v5 (4-class grocery detection)

Task type: 4-class object detection

Classes: Bottled Water, Noodles, canned-goods, rice

Input shape: RGB 640x640, scaled to [0, 1]

Output format: bounding boxes with class index and confidence

Deployment note: use TFLite model with NMS post-processing on Pi.
