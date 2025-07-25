"""Module to define CONSTANTS used across the DEEPaaS Interface.

This module is used to define CONSTANTS used across the AI-model package.
Do not misuse this module to define variables that are not CONSTANTS or
that are not used across the `api` package. You can use the `config.py`
file on your model package to define CONSTANTS related to your model.

By convention, the CONSTANTS defined in this module are in UPPER_CASE.
"""

import os
import logging
import ast
from importlib import metadata
from pathlib import Path
import datetime

MODEL_LIST = [
    "yolov8n.yaml",
    "yolov8n.pt",
    "yolov8s.yaml",
    "yolov8s.pt",
    "yolov8m.yaml",
    "yolov8m.pt",
    "yolov8l.yaml",
    "yolov8l.pt",
    "yolov8x.yaml",
    "yolov8x.pt",

    "yolov9c.yaml",
    "yolov9c.pt",
    "yolov9e.yaml",
    "yolov9e.pt",

    "yolo11n.yaml",
    "yolo11n.pt",
    "yolo11s.yaml",
    "yolo11s.pt",
    "yolo11m.yaml",
    "yolo11m.pt",
    "yolo11l.yaml",
    "yolo11l.pt",
    "yolo11x.yaml",
    "yolo11x.pt"
]

# Default AI model
MODEL_NAME = os.getenv("MODEL_NAME", 
                       default="socib_beach_wracks_identification_api")

# Get AI model metadata
MODEL_METADATA = metadata.metadata(MODEL_NAME)


# Fix metadata for authors from pyproject parsing
_EMAILS = MODEL_METADATA["Author-email"].split(", ")
_EMAILS = map(lambda s: s[:-1].split(" <"), _EMAILS)
MODEL_METADATA["Author-emails"] = dict(_EMAILS)

# Fix metadata for authors from pyproject parsing
_AUTHORS = MODEL_METADATA.get("Author", "").split(", ")
_AUTHORS = [] if _AUTHORS == [""] else _AUTHORS
_AUTHORS += MODEL_METADATA["Author-emails"].keys()
MODEL_METADATA["Authors"] = sorted(_AUTHORS)
# DEEPaaS can load more than one installed models. Therefore, in order to
# avoid conflicts, each default PATH environment variables should lead to
# a different folder. The current practice is to use the path from where the
# model source is located.
BASE_PATH = Path(__file__).resolve(strict=True).parents[1]

# Path definition for data folder
DATA_PATH = os.getenv("DATA_PATH", default=BASE_PATH / "data")
DATA_PATH = Path(DATA_PATH)
TEST_DATA_PATH = os.getenv(
    "TEST_DATA_PATH", default=BASE_PATH / "tests/data"
)
TEST_DATA_PATH = Path(TEST_DATA_PATH)
# Path definition for the pre-trained models
MODELS_PATH = os.getenv("MODELS_PATH", default=BASE_PATH / "models")
MODELS_PATH = Path(MODELS_PATH)

# Path definition for the default pre-trained models
DEFAULT_MODEL_PATH = MODELS_PATH

REMOTE_PATH = os.getenv("MODELS_PATH", default="models")
# logging level across API modules can be setup via API_LOG_LEVEL,
# options: DEBUG, INFO(default), WARNING, ERROR, CRITICAL
ENV_LOG_LEVEL = os.getenv("API_LOG_LEVEL", default="INFO")
LOG_LEVEL = getattr(logging, ENV_LOG_LEVEL.upper())


try:
    MODEL_LIST = os.getenv("MODEL_LIST", default=MODEL_LIST)
    if isinstance(MODEL_LIST, str):
        # Parse the string as a list of strings
        MODEL_LIST = ast.literal_eval(MODEL_LIST)
except KeyError as err:
    raise RuntimeError(
        "Undefined configuration for MODEL_LIST. "
    ) from err

# Specify that the default task is segmentation
YOLO_DEFAULT_TASK_TYPE = os.getenv(
    "YOLO_DEFAULT_TASK_TYPE", default="seg"
)


# Specify default timestamped weights for your trained models
# to be utilized during
# prediction. Format them as timestamp1, timestamp2, timestamp3, ...
YOLO_DEFAULT_WEIGHTS = os.getenv(
    "YOLO_DEFAULT_WEIGHTS", default=f'{DEFAULT_MODEL_PATH}/yolo11m_170325'
)
YOLO_DEFAULT_WEIGHTS = (
    YOLO_DEFAULT_WEIGHTS.split(",")
    if YOLO_DEFAULT_WEIGHTS
    else [None]
)


# Variables related to mlfow
try:
    MLFLOW_TRACKING_URI = os.getenv(
        "MLFLOW_TRACKING_URI",
        default="https://mlflow.cloud.ai4eosc.eu/",
    )
    MLFLOW_EXPERIMENT_NAME = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", default="socib_beach_wracks_identification"
    )
    MLFLOW_RUN = os.getenv(
        "MLFLOW_RUN",
        default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

except KeyError as err:
    raise RuntimeError(
        "Undefined configuration for mlflow settings"
    ) from err
