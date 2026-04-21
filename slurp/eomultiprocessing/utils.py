#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2022-2026 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of slurp
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

"""
This module contains some utils function for slurp processing.
"""

import os
from collections import namedtuple
from copy import deepcopy
from typing import Any, Dict, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

MpTile = namedtuple(
    "MpTile",
    [
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "top_margin",
        "right_margin",
        "left_margin",
        "bottom_margin",
        "height",
        "width",
        "height_margin",
        "width_margin",
    ],
)

MAX_CACHE = 64  # Mb, cache size for rasterio reading and writing operations,
# by default 5% of the usable physical RAM


def write(
    data: np.ndarray,
    img_path: str,
    target_profile: Dict[str, Any],
    binary: bool = False,
) -> None:
    """Save in file with rasterio."""
    with rasterio.Env(GDAL_CACHEMAX=MAX_CACHE):
        if binary:
            with rasterio.open(
                img_path, "w", nbits=1, **target_profile
            ) as out_dataset:
                out_dataset.write(data, indexes=1)
        else:
            with rasterio.open(img_path, "w", **target_profile) as out_dataset:
                out_dataset.write(data, indexes=1)


def write_window(
    img_buffer: np.ndarray,
    img_path: str,
    target_profile: Dict[str, Any],
    tile: MpTile,
    binary: bool = False,
) -> None:
    """Update window in file."""
    width = tile.end_x - tile.start_x + 1
    height = tile.end_y - tile.start_y + 1
    mode = "r+" if os.path.isfile(img_path) else "w"
    with rasterio.Env(GDAL_CACHEMAX=MAX_CACHE):
        if binary:
            with rasterio.open(
                img_path, mode, nbits=1, **target_profile
            ) as out_dataset:
                out_dataset.write(
                    img_buffer,
                    window=Window(tile.start_x, tile.start_y, width, height),
                    indexes=1,
                )
        else:
            with rasterio.open(img_path, mode, **target_profile) as out_dataset:
                out_dataset.write(
                    img_buffer,
                    window=Window(tile.start_x, tile.start_y, width, height),
                    indexes=1,
                )


def read(img_path: str) -> np.ndarray:
    """Read a file with minimal GDAL cache"""
    with rasterio.Env(GDAL_CACHEMAX=MAX_CACHE):
        with rasterio.open(img_path, "r") as src:
            data = src.read()

    return data


def read_and_get_profile(img_path: str) -> Tuple[np.ndarray, dict]:
    """Read a file with minimal GDAL cache and also get its profile"""
    with rasterio.Env(GDAL_CACHEMAX=MAX_CACHE):
        with rasterio.open(img_path, "r") as src:
            input_profile = src.profile.copy()
            data = src.read()

    return data, input_profile


def read_window(img_path: str, tile: MpTile) -> np.ndarray:
    """Read a window from a file with minimal GDAL cache"""
    col_off = tile.start_x - tile.left_margin
    row_off = tile.start_y - tile.top_margin
    with rasterio.Env(GDAL_CACHEMAX=MAX_CACHE):
        with rasterio.open(img_path, "r") as src:
            data = src.read(
                1,
                window=Window(
                    col_off, row_off, tile.width_margin, tile.height_margin
                ),
            )

    return data


def create_image(
    profile: Dict[str, Any],
    fill_value: Any = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create an empty image ndarray from an existing raster profile.

    The returned profile is identical (deep copy) to the input profile.

    Parameters
    ----------
    profile : dict
        Rasterio profile used as template.
    fill_value : Any, optional
        Initial value used to fill the image (default: 0).

    Returns
    -------
    Tuple[np.ndarray, dict]
        (image_array, copied_profile)

    Notes
    -----
    - Shape follows rasterio convention: (count, height, width)
    - dtype strictly follows profile["dtype"]
    """

    new_profile = deepcopy(profile)

    height = new_profile["height"]
    width = new_profile["width"]
    count = new_profile.get("count", 1)
    dtype = np.dtype(new_profile["dtype"])

    # Create ndarray
    image = np.full(
        (count, height, width),
        fill_value,
        dtype=dtype,
    )

    return image, new_profile
