"""Utilities module for API endpoints and methods.
This module is used to define API utilities and helper functions. You can
use and edit any of the defined functions to improve or add methods to
your API.

The module shows simple but efficient example utilities. However,
you may need to modify them for your needs.
"""

import logging
import os
import sys

from . import config

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def ls_dirs(path):
    """Utility to return a list of directories available in `path` folder.

    Arguments:
        path -- Directory path to scan for folders.

    Returns:
        A list of strings for found subdirectories.
    """
    logger.debug("Scanning directories at: %s", path)
    dirscan = (x.name for x in path.iterdir() if x.is_dir())
    return sorted(dirscan)


def generate_arguments(schema):
    """Function to generate arguments for DEEPaaS using schemas."""

    def arguments_function():  # fmt: skip
        logger.debug("Web args schema: %s", schema)
        return schema().fields

    return arguments_function


def predict_arguments(schema):
    """Decorator to inject schema as arguments to call predictions."""

    def inject_function_schema(func):
        get_args = generate_arguments(schema)
        sys.modules[func.__module__].get_predict_args = get_args
        return func  # Decorator that returns same function

    return inject_function_schema


def validate_and_modify_path(path, base_path):
    """
    Validate and modify a file path, ensuring it exists

    Args:
        path (str): The input file path to validate.
        base_path (str): The base path to join with 'path' if it
        doesn't exist as-is.

    Returns:
        str: The validated and possibly modified file path.
    """
    if not os.path.exists(path):
        path = os.path.join(base_path, path)
        if not os.path.exists(path):
            raise ValueError(
                f"The path {path} does not exist.Please provide a valid path."
            )
    return path


def generate_directory_tree(path):
    tree = {
        "name": os.path.basename(path),
        "type": "directory",
        "children": [],
    }

    if os.path.exists(path) and os.path.isdir(path):
        subdirectories = [
            d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
        ]
        subdirectories.sort()

        for subdir in subdirectories:
            subdir_path = os.path.join(path, subdir)
            tree["children"].append(generate_directory_tree(subdir_path))

    return tree
