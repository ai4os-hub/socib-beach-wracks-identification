"""
Package to create dataset, build training and prediction pipelines.

This file should define or import all the functions needed to operate the
methods defined at socib_beach_wracks_identification/api.py. Complete the TODOs
with your own code or replace them importing your own functions.
For example:
```py
from your_module import your_function as predict
from your_module import your_function as training
```
"""

import logging
import os

import cv2
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

import socib_beach_wracks_identification.config as cfg
import socib_beach_wracks_identification.sahi_utils as sahi_utils
import socib_beach_wracks_identification.utils as utils

logger = logging.getLogger(__name__)
logger.setLevel(cfg.LOG_LEVEL)


# TODO: warm (Start Up)
# = HAVE TO MODIFY FOR YOUR NEEDS =
def warm(
    **kwargs,
):
    """Main/public method to start up the model"""
    # if necessary, start the model
    pass


def predict(
    **args,
):
    """Main/public method to perform prediction"""
    print("arg of prediction are", args)

    image_size = args.get("imgsz", [640, 480])
    if len(image_size) != 2:
        raise ValueError(
            "For SAHI inference, please provide image size as "
            "a list of two integers: [slice_height, slice_width]."
        )

    if args.get("sahi", False):
        return predict_sahi(**args)

    model = YOLO(args["model"])
    test_image_path = args["files"]
    results = []
    for image_path in test_image_path:
        print(
            "Evaluating:",
            image_path,
        )
        utils.remove_keys_from_dict(
            args,
            ["files", "accept", "task_type", "sahi"],
        )
        print("is torch cuda available?", torch.cuda.is_available())
        result = model.predict(
            image_path,
            **args,
        )
        logger.debug(f"[predict()]: {result}")
        results.append(result)
    return results[0]


def predict_sahi(**args):
    """Main/public method to perform prediction with SAHI"""

    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args["model"],
        confidence_threshold=args["conf"],
        device=device,
    )

    sahi_result = get_sliced_prediction(
        image=args["files"][0],
        detection_model=detection_model,
        slice_height=args["imgsz"][0],
        slice_width=args["imgsz"][1],
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    print(f"SAHI detected {len(sahi_result.object_prediction_list)} objects")

    original_image_path = args["files"][0]
    original_image = cv2.imread(original_image_path)  # BGR format

    results = sahi_utils.SahiCustomResult(sahi_result, original_image)

    print("SAHI Custom Prediction results:", results)
    return [results]


if __name__ == "__main__":
    args = {
        "files": ["tests/data/seg/images/test/clm_s_03_2015-05-10-14-00.png"],
        "model": os.getenv("YOLO_DEFAULT_WEIGHTS"),
        "imgsz": [704, 512],
        "conf": 0.25,
        "iou": 0.5,
        "show_labels": True,
        "show_conf": True,
        "augment": True,
        "classes": None,
        "show_boxes": True,
    }
    predict(**args)
