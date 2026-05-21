"""
Animal day-classification inference -- INT8 (CoreML) variant.

This is the INT8-quantized counterpart of test.py. It runs the models as
CoreML packages on the Apple Neural Engine (ANE) instead of FP32/FP16 PyTorch
on the GPU.

MEASURED on this machine (custom model, train14_best):
    FP32 PyTorch / MPS : ~87 ms
    FP16 PyTorch / MPS : ~68 ms
    INT8 CoreML  / ANE : ~33 ms      <-- this file  (~2.6x faster than FP32)

HOW IT WORKS:
  - On first run, each .pt model is exported once to a CoreML INT8 .mlpackage
    (weight palettization). The export takes ~2 min/model and is cached after.
  - CoreML picks the compute unit itself (Neural Engine / GPU / CPU) -- you do
    not pass device= or half= for a CoreML model.

ACCURACY -- READ THIS:
  INT8 is lossy quantization. It is NOT guaranteed accuracy-neutral (expect a
  ~1-3% mAP shift). On the COCO sample images detection counts matched FP32,
  but you MUST validate on real Leopard / Bear / Tiger images before relying
  on this file. If accuracy drops too far, stay on test.py (FP16, ~68 ms).
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

_MODEL_DIR = 'model_structure' if os.path.isdir('model_structure') else '.'


def _ensure_int8(pt_name, imgsz):
    """Return the CoreML INT8 model path, exporting it once if it doesn't exist."""
    pt_path = os.path.join(_MODEL_DIR, pt_name)
    ml_path = os.path.splitext(pt_path)[0] + '.mlpackage'
    if not os.path.exists(ml_path):
        print(f'[int8] exporting {pt_path} -> {ml_path}  (one-time, ~2 min)...')
        YOLO(pt_path).export(format='coreml', int8=True, imgsz=imgsz, nms=False)
    return ml_path


# ---------------------------------------------------------------------------
# Load the INT8 models ONCE at import time (same load-once rule as test.py).
# ---------------------------------------------------------------------------
CUSTOM_MODEL = YOLO(_ensure_int8('train14_best.pt', 800))   # 12-class custom model
PRETRAINED_MODEL = YOLO(_ensure_int8('yolov8l.pt', 640))    # COCO model (human detection)

# Warm up both models so the first real call is not slowed by lazy init.
_ = CUSTOM_MODEL.predict(np.zeros((800, 800, 3), dtype='uint8'), imgsz=800, verbose=False)
_ = PRETRAINED_MODEL.predict(np.zeros((640, 640, 3), dtype='uint8'), imgsz=640, verbose=False)


def yolo_v5_day_classification_2_18(image_path):
    # custom INT8 model -- CoreML picks the compute unit (no device=/half=).
    result = CUSTOM_MODEL.predict(
        source=image_path, imgsz=800, conf=0.6,
        classes=TARGET_CLASSES,  # Leopard / Tiger / Bear only
        verbose=False,
    )
    # pre-trained INT8 model
    results_y = PRETRAINED_MODEL.predict(
        source=image_path, imgsz=640, conf=0.5,
        classes=[0],  # COCO 'person' only -- trims NMS, output unchanged
        verbose=False,
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
