"""Endpoint functions to integrate your model with the DEEPaaS API.

For more information about how to edit the module see, take a look at the
docs [1] and at a canonical exemplar module [2].

[1]: https://docs.ai4os.eu/
[2]: https://github.com/ai4os-hub/ai4os-demo-app
"""

import argparse
import datetime
import json
import logging
import os
import shutil
import tempfile

import torch
from aiohttp.web import HTTPException
from deepaas.model.v2.wrapper import UploadedFile
from ultralytics import YOLO, settings

import socib_beach_wracks_identification as aimodel
from socib_beach_wracks_identification.utils import (
    mlflow_fetch,
    mlflow_logging,
)

from . import config, responses, schemas, utils

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


# global var

MLFLOW_MODEL_NAME = "socib_beach_wracks_identification"


def get_metadata():
    """Returns a dictionary containing metadata information about the module.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    try:  # Call your AI model metadata() method
        logger.info("Collecting metadata from: %s", config.MODEL_NAME)
        print(config.BASE_PATH)
        metadata = {
            "author": config.MODEL_METADATA.get("authors"),
            "author-email": config.MODEL_METADATA.get("author-emails"),
            "description": config.MODEL_METADATA.get("summary"),
            "license": config.MODEL_METADATA.get("license"),
            "version": config.MODEL_METADATA.get("version"),
            "models_local": utils.ls_dirs(config.MODELS_PATH),
            # "models_remote": utils.ls_remote(),
            "datasets": utils.generate_directory_tree(config.DATA_PATH),
        }
        logger.debug("Package model metadata: %s", metadata)
        return metadata
    except Exception as err:
        raise HTTPException(reason=err) from err


@utils.predict_arguments(schema=schemas.PredArgsSchema)
def predict(**args):
    """Performs model prediction from given input data and parameters.

    Arguments:
            **args -- Arbitrary keyword arguments from PredArgsSchema.

    Raises:
            HTTPException: Unexpected errors aim to return 50X

    Returns:
            The predicted model values json, png, pdf or mp4 file.
    """

    logger.debug("Predict with args: %s", args)
    try:
        if args["model"] is None:
            args["model"] = config.DEFAULT_MODEL_PATH  # Only seg is enabled

        else:
            path = os.path.join(args["model"], "weights/best.pt")
            args["model"] = utils.validate_and_modify_path(path, config.MODELS_PATH)

        with tempfile.TemporaryDirectory() as tmpdir:
            for f in [args["files"]]:
                shutil.copy(
                    f.filename,
                    tmpdir + "/" + os.path.basename(f.original_filename),  # nolint
                )

            args["files"] = [os.path.join(tmpdir, t) for t in os.listdir(tmpdir)]
            result = aimodel.predict(**args)
            logger.debug("Predict result: %s", result)
            logger.info("Returning content_type for: %s", args["accept"])
            return responses.response_parsers[args["accept"]](result, **args)

    except Exception as err:
        raise HTTPException(reason=err) from err


def main():
    """
    Runs above-described methods from CLI
    uses: python3 path/to/api/__init__.py method --arg1 ARG1_VALUE
     --arg2 ARG2_VALUE
    """
    method_dispatch = {
        "get_metadata": get_metadata,
        "predict": predict,
    }

    chosen_method = args.method
    logger.debug("Calling method: %s", chosen_method)
    if chosen_method in method_dispatch:
        method_function = method_dispatch[chosen_method]

        if chosen_method == "get_metadata":
            results = method_function()
        else:
            logger.debug("Calling method with args: %s", args)
            del vars(args)["method"]
            if hasattr(args, "files"):
                file_extension = os.path.splitext(args.files)[1]
                args.files = UploadedFile(
                    "files",
                    args.files,
                    "application/octet-stream",
                    f"files{file_extension}",
                )
            results = method_function(**vars(args))
        print(json.dumps(results))
        logger.debug("Results: %s", results)
        return results
    else:
        print("Invalid method specified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model parameters", add_help=False)
    cmd_parser = argparse.ArgumentParser()
    subparsers = cmd_parser.add_subparsers(
        help='methods. Use "api.py method --help" to get more info',
        dest="method",
    )
    get_metadata_parser = subparsers.add_parser(
        "get_metadata", help="get_metadata method", parents=[parser]
    )

    predict_parser = subparsers.add_parser(
        "predict", help="commands for prediction", parents=[parser]
    )

    utils.add_arguments_from_schema(schemas.PredArgsSchema(), predict_parser)

    # train_parser = subparsers.add_parser(
    #     "train", help="commands for training", parents=[parser]
    # )

    # utils.add_arguments_from_schema(schemas.TrainArgsSchema(), train_parser)

    args = cmd_parser.parse_args()

    main()

    """
    python3 api/__init__.py  train --model yolov8n.yaml\
    --task_type  det\
    --data /srv/football-players-detection-7/data.yaml\
    --Enable_MLFLOW --epochs 50
    python3 api/__init__.py  predict --files \
    /srv/yolov8_api/tests/data/det/test/cat1.jpg\
    --task_type  det --accept application/json
    """
