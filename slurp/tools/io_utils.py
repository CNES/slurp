#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of SLURP
# (see https://github.com/CNES/slurp).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Brings together useful functions common to the different scripts"""
import json

import numpy as np
import logging
import rasterio as rio
from slurp.tools.pydantic_class import load_config, MainConfig

from slurp.tools.constant import COMPRESSION, DRIVER

logger = logging.getLogger("slurp")

def read_json(
    main_config_file: str, keys: list
) -> dict:
    """
    Read JSON config files

    :param str main_config_file: Path to the main JSON config file
    :param list keys: Keys to read in the JSON files
    :param str user_config_file: Path to the overload JSON config file (None by default)
    :returns: dictionary of arguments
    """
    # Read the JSON data from the main config
    try:
        config = load_config(main_config_file, MainConfig)
        full_args = config.model_dump()
        argsdict = full_args[keys[0]]
        for key in keys[1:]:
            argsdict.update(full_args[key])

    except FileNotFoundError:
        logger.error(f"File {main_config_file} not found.")
    except json.JSONDecodeError:
        logger.error(
            f"Error decoding JSON data from {main_config_file}. Please check the file format."
        )

    return argsdict


def save_image(
    image,
    file,
    crs=None,
    transform=None,
    nodata=None,
    rpc=None,
    colormap=None,
    tags=None,
    **kwargs,
):
    """
    Save 1 band numpy image to file with deflate compression.
    Note that rio.dtype is string so convert np.dtype to string.
    rpc must be a dictionary.
    """

    dataset = rio.open(
        file,
        "w",
        driver=DRIVER,
        compress=COMPRESSION.lower(),
        height=image.shape[0],
        width=image.shape[1],
        count=1,
        dtype=str(image.dtype),
        crs=crs,
        transform=transform,
        **kwargs,
    )
    dataset.write(image, 1)
    dataset.nodata = nodata

    if rpc:
        dataset.update_tags(**rpc, ns="RPC")

    if colormap:
        dataset.write_colormap(1, colormap)

    if tags:
        dataset.update_tags(**tags)

    dataset.close()
    del dataset
