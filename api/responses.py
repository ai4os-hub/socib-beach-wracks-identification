"""
Module for defining custom API response parsers and content types.
This module is used by the API server to convert the output of the
requested method into the desired format.
"""

import json
import logging
import os
import tempfile
from io import BytesIO

import cv2
from PIL import Image
from PyPDF3 import PdfFileMerger
import numpy as np

from . import config

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def json_response(results, **options):
    """Converts the prediction or training results into JSON format.

    Arguments:
        results -- Result value from call, expected as a list or dict
        options -- Additional options (e.g., task type).

    Raises:
        RuntimeError: Unsupported response type.

    Returns:
        JSON data with a clean structure.
    """
    result_data = []
    logger.debug("Response result type: %s", type(results))
    logger.debug("Response result: %s", results)
    logger.debug("Response options: %s", options)

    try:
        for element in results:
            # Use the proper `to_json` method to serialize each result
            prediction = (
                element.to_json() if hasattr(element, "to_json") else element
            )
            if isinstance(prediction, str):
                # Convert stringified JSON to actual JSON
                prediction = json.loads(prediction)
            result_data.append(prediction)
        return result_data

    except Exception as err:
        logger.warning("Error converting result to JSON: %s", err)
        raise RuntimeError("Unsupported response type") from err


def pdf_response(results, **options):
    """Converts the prediction or training results into pdf return format.

    Arguments:
        result -- Result value from call, expected dict
        options -- Not used, added for illustration purpose.

    Raises:
        RuntimeError: Unsupported response type.

    Returns:
        Converted result into pdf buffer format.
    """
    logger.debug("Response result type: %d", type(results))
    logger.debug("Response result: %d", results)
    logger.debug("Response options: %d", options)

    try:
        merger = PdfFileMerger()
        for element in results:
            plot_array = element.plot(
                labels=options["show_labels"],
                conf=options["show_conf"],
                boxes=options["show_boxes"],
            )
            plot_array = plot_array[..., ::-1]

            im = Image.fromarray(plot_array)
            im = im.convert("RGB")

            buffer = BytesIO()
            buffer.name = "output.pdf"
            im.save(buffer)
            merger.append(buffer)
            buffer.seek(0)
        buffer_out = BytesIO()
        merger.write(buffer_out)
        buffer_out.name = "output.pdf"
        buffer_out.seek(0)
        return buffer_out
    except Exception as err:  # TODO: Fix to specific exception
        logger.warning("Error converting result to pdf: %s", err)
        raise RuntimeError("Unsupported response type") from err


def png_response(results, **options):
    logger.debug("Response result type: %d", type(results))
    logger.debug("Response result: %d", results)
    logger.debug("Response options: %d", options)
    try:
        for result in results:
            # this will return a numpy array with the labels
            result = result.plot(
                labels=options["show_labels"],
                conf=options["show_conf"],
                boxes=options["show_boxes"],
                font_size=6.0,
            )
            success, buffer = cv2.imencode(".png", result)
            if not success:
                return "Error encoding image", 500

            # Create a BytesIO object and write the buffer into it
            image_buffer = BytesIO(buffer)

        return image_buffer
    except Exception as err:  # TODO: Fix to specific exception
        logger.warning("Error converting result to png: %s", err)
        raise RuntimeError("Unsupported response type") from err


def mask_response(results, **options):
    logger.debug("Response result type: %s", type(results))
    logger.debug("Response options: %s", options)

    CLASS_TO_GRAY = {
        0: 255,  # White for dense wracks
        1: 128,  # Gray for intermediate wracks
    }

    try:
        for result in results:
            if result.masks is not None:
                # Get masks shape (N, H, W) and class IDs shape (N,)
                masks = result.masks.data.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                # Create an empty single-channel image (grayscale) for the background (0)
                H, W = masks.shape[1:]
                combined_mask = np.zeros((H, W), dtype=np.uint8)

                # Iterate over each detected mask
                for i, mask in enumerate(masks):
                    cls_id = class_ids[i]

                    # Get the fixed grayscale value from the dictionary
                    # Default to 0 (background) if the class ID is not found
                    gray_value = CLASS_TO_GRAY.get(cls_id, 0)

                    # Apply the gray value only where the mask confidence is > 0.5
                    # and only if the gray_value is greater than 0
                    if gray_value > 0:
                        combined_mask = np.where(
                            mask > 0.5, gray_value, combined_mask
                        )

                # Resize the final mask to match the original input image size
                h_orig, w_orig = result.orig_shape

                # INTER_NEAREST is strictly required to prevent smoothing/blurring at the edges
                final_image = cv2.resize(
                    combined_mask,
                    (w_orig, h_orig),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                h_orig, w_orig = result.orig_shape
                final_image = np.zeros((h_orig, w_orig), dtype=np.uint8)

            success, buffer = cv2.imencode(".png", final_image)
            if not success:
                return "Error encoding image", 500

            # Create a BytesIO object and write the buffer into it
            image_buffer = BytesIO(buffer)

        return image_buffer
    except Exception as err:
        logger.warning("Error converting result to mask png: %s", err)
        raise RuntimeError("Unsupported response type") from err


def create_video_in_buffer(frame_arrays, output_format="mp4"):
    height, width, _ = frame_arrays[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    with tempfile.NamedTemporaryFile(
        suffix="." + output_format, delete=False
    ) as temp_file:
        temp_filename = temp_file.name
        out = cv2.VideoWriter(temp_filename, fourcc, 20.0, (width, height))

        for frame in frame_arrays:
            out.write(frame)

        out.release()

    final_filename = "output.mp4"
    os.rename(temp_filename, final_filename)
    # Open the renamed file for reading
    message = open(final_filename, "rb")
    return message


def mp4_response(results, **options):
    """Converts the prediction or training results into
    mp4 return format.

    Arguments:
        result -- Result value from call, expected either dict or str
        options -- Not used, added for illustration purpose.

    Raises:
        RuntimeError: Unsupported response type.

    Returns:
        Converted result into mp4 buffer format.
    """
    # Process MP4 video response
    logger.debug("Response result type: %d", type(results))
    logger.debug("Response result: %d", results)
    logger.debug("Response options: %d", options)
    new_results = []
    for result in results[0]:
        # this will return a numpy array with the labels
        new_results.append(
            result.plot(
                labels=options["show_labels"],
                conf=options["show_conf"],
                boxes=options["show_boxes"],
            )
        )
    message = create_video_in_buffer(new_results)
    return message


response_parsers = {
    "application/json": json_response,
    # "application/pdf": pdf_response,
    "image/png": png_response,
    "image/mask": mask_response,
    # "video/mp4": mp4_response,
}
content_types = list(response_parsers)
