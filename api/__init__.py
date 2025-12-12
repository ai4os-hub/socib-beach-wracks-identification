"""Endpoint functions to integrate your model with the DEEPaaS API.

For more information about how to edit the module see, take a look at the
docs [1] and at a canonical exemplar module [2].

[1]: https://docs.ai4os.eu/
[2]: https://github.com/ai4os-hub/ai4os-demo-app
"""

import logging
import os
import shutil
import tempfile

from aiohttp.web import HTTPException

import socib_beach_wracks_identification as aimodel

from . import config, responses, schemas, utils

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


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
            args["model"] = utils.validate_and_modify_path(
                path, config.MODELS_PATH
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            for f in [args["files"]]:
                shutil.copy(
                    f.filename,
                    tmpdir
                    + "/"
                    + os.path.basename(f.original_filename),  # nolint
                )

            args["files"] = [
                os.path.join(tmpdir, t) for t in os.listdir(tmpdir)
            ]
            result = aimodel.predict(**args)
            logger.debug("Predict result: %s", result)
            logger.info("Returning content_type for: %s", args["accept"])
            return responses.response_parsers[args["accept"]](result, **args)

    except Exception as err:
        raise HTTPException(reason=err) from err
