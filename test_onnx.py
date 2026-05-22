"""
Animal day-classification inference -- cross-platform ONNX variant.

This is the portable counterpart of test.py / test_int8.py. It runs the custom
model from train14_best_int8.onnx via ONNX Runtime, which works on
Windows, Linux, AND macOS -- unlike the CoreML .mlpackage (Mac only).

  Animal model : train14_best_int8.onnx  -- INT8 backbone + FP32 detection head.
                 Verified 100% species agreement with the FP32 model; boxes
                 correct (the ONNX path is not affected by the .pt box bug).
  Human model  : yolov8l.pt              -- ultralytics auto-downloads it on
                 first run; .pt runs on any OS via torch.

Requirements:  pip install ultralytics onnxruntime
  (onnxruntime is CPU by default; install onnxruntime-gpu for NVIDIA CUDA.)
"""

import os
import numpy as np
import torch
from ultralytics import YOLO

# check_human_detection and pre_train_human_class_mapping are defined elsewhere
# in your project -- import them here exactly as in test.py, e.g.:
# from helpers import check_human_detection, pre_train_human_class_mapping

# Detect ONLY these classes from the custom model: Leopard(0), Tiger(6), Bear(10).
TARGET_CLASSES = [0, 6, 10]

# Device for the .pt human model. The ONNX model uses ONNX Runtime's own
# execution provider, so it does not take a device argument.
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

_MODEL_DIR = 'model_structure' if os.path.isdir('model_structure') else '.'

# ---------------------------------------------------------------------------
# Load both models ONCE at import time (same load-once rule as test.py).
# ---------------------------------------------------------------------------
CUSTOM_MODEL = YOLO(os.path.join(_MODEL_DIR, 'train14_best_int8.onnx'))  # cross-platform INT8
PRETRAINED_MODEL = YOLO(os.path.join(_MODEL_DIR, 'yolov8l.pt'))          # human detection

# Warm up both models so the first real call is not slowed by lazy init.
_ = CUSTOM_MODEL.predict(np.zeros((800, 800, 3), dtype='uint8'), imgsz=800, verbose=False)
_ = PRETRAINED_MODEL.predict(np.zeros((640, 640, 3), dtype='uint8'), imgsz=640,
                             device=DEVICE, verbose=False)


def yolo_v5_day_classification_2_18(image_path):
    # custom model -- ONNX Runtime (no device= argument)
    result = CUSTOM_MODEL.predict(
        source=image_path, imgsz=800, conf=0.6,
        classes=TARGET_CLASSES,  # Leopard / Tiger / Bear only
        verbose=False,
    )
    # pre-trained model -- human detection
    results_y = PRETRAINED_MODEL.predict(
        source=image_path, imgsz=640, conf=0.5,
        classes=[0],  # COCO 'person' only -- trims NMS, output unchanged
        device=DEVICE, verbose=False,
    )
    custom_output_classes = result[0].boxes.cls
    custom_confidence_values = result[0].boxes.conf

    # print("Pre-trained model") print("Pre-traind pretrained_output classes and conf ")
    pretrained_output_classes = results_y[0].boxes.cls
    pretrained_confidence_values = results_y[0].boxes.conf

    #counting detection in both models
    detection_count = int(len(result[0].boxes.cls))
    detection_count_y = int(len(results_y[0].boxes.cls))

     #species mapping for pre-trianed model
    species_mapping = {
        0: 'Leopard',
        1: 'Cat',
        2: 'Dog',
        3: 'Deer',
        4: 'Goat',
        5: 'Monkey',
        6: 'Tiger',
        7: 'Wild_boar',
        8: 'Cow',
        9: 'Hen',
        10: 'Bear',
        11: 'Byson',
    }
    #intalizing main output which we'll return from function
    main_output_class_species = 'Nothing'
    main_conf_pred = 0.0
    for i in range(len(result)):
        class_pred = result[i].boxes.cls
        conf_pred = result[i].boxes.conf
        class_pred_y = results_y[i].boxes.cls
        conf_pred_y = results_y[i].boxes.conf
        #No detection in custom and pre-trained model
        if detection_count == 0 and detection_count_y == 0:

            return 0.0, 'Nothing'
        #detection in custom model but no detection in pre-trained model
        elif detection_count !=0 and detection_count_y == 0:
            max_index = torch.argmax(conf_pred).item()
            output_class = int(class_pred[max_index].item())
            main_output_class_species = species_mapping[output_class]
            main_conf_pred = conf_pred[max_index].item()
            return main_conf_pred, main_output_class_species
        #no detection in custom model but detection in pre-trained model
        elif detection_count == 0 and detection_count_y != 0:
            pretrained_detection = check_human_detection(pretrained_output_classes, pretrained_confidence_values, pre_train_human_class_mapping)
            if pretrained_detection:
                return pretrained_detection[1].item(), pretrained_detection[0]
            else:
                return 0.0, 'Nothing'
        #detection in both custom and pre-trained model
        elif detection_count != 0 and detection_count_y != 0:
            #check human detection
            pretrained_detection = check_human_detection(pretrained_output_classes, pretrained_confidence_values, pre_train_human_class_mapping)
            if pretrained_detection:
                return pretrained_detection[1].item(), pretrained_detection[0]
            if detection_count == 1:
                return custom_confidence_values.item(), species_mapping[custom_output_classes.item()]
            # Case when more than one detection is present
            elif detection_count > 1:
              # The model is restricted to Leopard / Tiger / Bear, so every
              # detection is a target animal -- return the most confident one.
              detections = [
                  (species_mapping[int(cls)], conf)
                  for cls, conf in zip(custom_output_classes, custom_confidence_values)
              ]
              most_confident = max(detections, key=lambda x: x[1])
              return most_confident[1].item(), most_confident[0]  # Confidence and species

    return main_conf_pred, main_output_class_species
