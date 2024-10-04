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


""" Brings together the auxiliary files reading functions """
from os import path

import rasterio as rio
import numpy as np
import scipy
from slurp.prepare import geometry
from slurp.tools.constant import NODATA_int16


def aux_file_recovery(file_ref: str, global_data: str, reprojected_data: str, sensor_mode: bool, dtm_file: str = "", geoid_file: str = ""):
    """
    Recover Global Data and reproject it into input reference image geometry (geo or sensor)

    :param str file_ref: path to the input reference image
    :param str global_data: path to the input Pekel global image (tile or .vrt)
    :param str reprojected_data: path for the recovered Pekel image
    :param bool sensor_mode: true if file_ref is in sensor mode (not georeferenced)
    :returns: global data cropped onto target image geometry
    """
    print(f"Recover file {global_data=} to {reprojected_data=} onto {file_ref=} geometry")
    if sensor_mode:
        geometry.sensor_projection(global_data, file_ref, dtm_file, geoid_file, reprojected_data)
    else:
        geometry.superimpose(
            global_data,
            file_ref,
            reprojected_data
        )
                

def std_convoluted(im: np.ndarray, N: int, min_value: float, max_value: float) -> np.ndarray:
    """
    Calculate the std of each pixel
    Based on a convolution with a kernel of 1 (size of the kernel given)

    :param np.ndarray im: input image
    :param int N: radius of kernel
    :param float min_value: min value of the input image
    :param float max_value: max value of the input image
    :returns: texture image
    """
    im2 = im ** 2
    kernel = np.ones((2 * N + 1, 2 * N + 1))
    ns = kernel.size * np.ones(im.shape)

    # Local mean with convolution
    s = scipy.signal.convolve2d(im, kernel, mode="same", boundary="symm")
    # local mean of the squared image with convolution
    s2 = scipy.signal.convolve2d(im2, kernel, mode="same", boundary="symm")

    # Invalid values will be handled later
    np.seterr(divide="ignore", invalid="ignore")
    res = np.sqrt((s2 - s ** 2 / ns) / ns)  # std calculation

    # Normalization
    res = 1000 * res / (max_value - min_value)

    res = np.where(np.isnan(res), 0, res)

    return res


def texture_task(input_buffers: list, input_profiles: list, params: dict) -> np.ndarray:
    """
    Compute textures

    :param list input_buffers: [im_vhr, valid_stack]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments, must contain the keys "nir", "texture_rad", "min_value" and "max_value"
    :returns: texture image
    """
    masked_band = np.ma.array(input_buffers[0][params["nir"] - 1], mask=np.logical_not(input_buffers[1]))
    texture = std_convoluted(masked_band.astype(float), params["texture_rad"], params["min_value"], params["max_value"])
    texture[np.logical_not(input_buffers[1][0])] = NODATA_int16

    return texture
