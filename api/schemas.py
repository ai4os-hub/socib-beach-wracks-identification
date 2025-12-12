"""Module for defining custom web fields to use on the API interface.
This module is used by the API server to generate the input form for the
prediction and training methods. You can use any of the defined schemas
to add new inputs to your API.

The module shows simple but efficient example schemas. However, you may
need to modify them for your needs.
"""

import marshmallow
from webargs import ValidationError, fields, validate

from . import config, responses, utils


class ModelName(fields.String):
    """Field that takes a string and validates against current available
    models at config.MODELS_PATH.
    """

    def _deserialize(self, value, attr, data, **kwargs):
        if value not in utils.ls_dir(config.MODELS_PATH):
            raise ValidationError(f"Checkpoint `{value}` not found.")
        return str(config.MODELS_PATH / value)


class Dataset(fields.String):
    """Field that takes a string and validates against current available
    data files at config.DATA_PATH.
    """

    def _deserialize(self, value, attr, data, **kwargs):
        if value not in utils.ls_dir(config.DATA_PATH / "processed"):
            raise ValidationError(f"Dataset `{value}` not found.")
        return str(config.DATA_PATH / "processed" / value)


class PredArgsSchema(marshmallow.Schema):
    class Meta:
        ordered = True

    files = fields.Field(
        required=True,
        type="file",
        location="form",
        metadata={
            "description": "Input an image or Video.\n"
            "accepted image formats: .bmo, .dng, .jpg, .jpeg, "
            ".mpo, .png, .tif, .tiff, .pfm, and .webp. \n"
            "accepted video formats: .asf, .avi, .gif, .m4v, .mkv,"
            ".mov, .mp4, .mpeg, .mpg, .ts, .wmv, .webm",
        },
    )

    model = fields.Str(
        metadata={
            "description": "The timestamp inside the 'models' directory "
            "indicates the time when you saved your trained model, "
            "The directory structure should resemble "
            "'models/your_timestamp/weights/best.pt'. "
            "To see the available timestamp, please run the "
            "get_metadata function and check model_local. "
            "If not provided, either a model from the MLflow "
            "registry will be loaded (if mlflow_fetch=true) "
            "or the pre-trained default model will be loaded "
            "depending on the task type."
        },
        load_default=config.YOLO_DEFAULT_WEIGHTS[0],
    )

    sahi = fields.Boolean(
        metadata={
            "description": "Whether to use Sahi for large image "
            "inference by splitting the image into smaller "
            "overlapping patches. Useful for high-resolution "
            "images where objects may be small relative to "
            "the overall image size."
        },
        load_default=False,
    )

    task_type = fields.Str(
        metadata={
            "description": "The type of task for load the pretrained model.\n"
            'The one available is "seg", for instance segmentation',
            "enum": config.YOLO_DEFAULT_TASK_TYPE,
        },
        load_default=config.YOLO_DEFAULT_TASK_TYPE,
    )

    imgsz = fields.List(
        fields.Int(),
        validate=validate.Length(max=2),
        metadata={
            "description": "image size as scalar or (h, w) list,"
            " i.e. (640, 480). Note: must be multiple of max stride 32"
        },
        load_default=[640, 480],
    )

    conf = fields.Float(
        metadata={"description": "object confidence threshold for detection"},
        load_default=0.25,
    )

    iou = fields.Float(
        metadata={
            "description": "intersection over union (IoU) threshold for NMS",
        },
        load_default=0.5,
    )

    show_labels = fields.Boolean(
        metadata={
            "description": "Show object labels in plots",
        },
        load_default=True,
    )
    show_conf = fields.Boolean(
        metadata={
            "description": "Show object confidence scores in plots."
            "if show_labels is False, show_conf is also False",
        },
        load_default=True,
    )

    # augment = fields.Boolean(
    #    metadata={
    #        "description": "Apply image augmentation to prediction sources. "
    #        "augment for segmentation has not supported yet.",
    #    },
    #    load_default=False,
    # )

    classes = fields.List(
        fields.Int(),
        metadata={
            "description": "Filter results by class, i.e. class=0, "
            "or class=[0,2,3]. Only detections belonging to the "
            "specified classes will be returned. Useful for focusing"
            " on relevant objects in multi-class detection."
        },
        load_default=None,
    )

    show_boxes = fields.Boolean(
        metadata={"description": "Show boxes in segmentation predictions"},
        load_default=True,
    )
    accept = fields.String(
        metadata={
            "description": "Return format for method response.",
            "location": "headers",
        },
        required=False,
        load_default="application/json",
        validate=validate.OneOf(responses.content_types),
    )
