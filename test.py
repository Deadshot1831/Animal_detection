"""
Animal day-classification inference (FP32 -- the accuracy reference).

PERFORMANCE NOTES:
  1. Models are loaded ONCE at import time, not on every call. Previously each
     call reloaded ~107 MB + ~88 MB of weights from disk -- this was the single
     biggest cost.
  2. Inference runs on the Apple-Silicon GPU (MPS) when available.
  3. Models are warmed up once so the first real request is not penalised.
  4. verbose=False removes per-call console logging overhead.

  None of these changes affect detection accuracy.

  FP16 note: half precision was tested and gives NO speedup on MPS -- ultralytics
  ignores half= on the MPS backend (verified bit-identical output over 196
  images). It is therefore not used here.

  For a ~2x faster INT8 variant (small accuracy trade-off), see test_int8.py.
"""

import os
import torch
from ultralytics import YOLO

# check_human_detection and pre_train_human_class_mapping are defined elsewhere
# in your project -- import them here exactly as you do today, e.g.:
# from helpers import check_human_detection, pre_train_human_class_mapping

# ---------------------------------------------------------------------------
# Load models ONCE at import time (previously reloaded inside the function).
# ---------------------------------------------------------------------------
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Detect ONLY these classes from the custom model: Leopard(0), Tiger(6), Bear(10).
TARGET_CLASSES = [0, 6, 10]

_MODEL_DIR = 'model_structure' if os.path.isdir('model_structure') else '.'

CUSTOM_MODEL = YOLO(os.path.join(_MODEL_DIR, 'train14_best.pt'))   # 12-class custom model
PRETRAINED_MODEL = YOLO(os.path.join(_MODEL_DIR, 'yolov8l.pt'))    # COCO model (human detection)

# Warm up both models so the first real call is not slowed by lazy init.
_ = CUSTOM_MODEL.predict(torch.zeros(1, 3, 800, 800), device=DEVICE, verbose=False)
_ = PRETRAINED_MODEL.predict(torch.zeros(1, 3, 640, 640), device=DEVICE, verbose=False)


def yolo_v5_day_classification_2_18(image_path):
    # custom model -> uses preloaded global model (no reload, runs on GPU)
    result = CUSTOM_MODEL.predict(
        source=image_path, imgsz=800, conf=0.6,
        classes=TARGET_CLASSES,  # Leopard / Tiger / Bear only
        device=DEVICE, verbose=False,
    )
    # pre-trained model -> uses preloaded global model
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
