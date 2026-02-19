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


"""Function to compute primitives"""

import numpy as np
import scipy

from slurp.tools.constant import NODATA_INT16


def compute_ndxi(
    input_buffer: list, input_profiles: list, params: dict
) -> np.ndarray:
    """
    Compute Normalize Difference X Index.
    Rescale to [-1000, 1000] int16 with nodata value = 32767
    1000 * (im_b1 - im_b2) / (im_b1 + im_b2)

    :param list input_buffer: VHR input image [im_vhr, valid_stack]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments, must contain the keys "im_b1" and "im_b2"
    :returns: NDXI
    """
    np.seterr(divide="ignore", invalid="ignore")
    im_ndxi = 1000.0 - (
        2000.0 * np.float32(input_buffer[0][params["im_b2"] - 1])
    ) / (
        np.float32(input_buffer[0][params["im_b1"] - 1])
        + np.float32(input_buffer[0][params["im_b2"] - 1])
    )
    # Special case where reflectance values are negative : we could obtain a NDVI slightly below -1
    # ex : R: 90.07, NIR: -1.24 --> NDVI: -1.02
    # In that special case, we prefer set the value to -1 or 1.
    # Otherwise we should modify the validity mask, to avoid these values to be taken into account
    # in the next steps of the classification algorithms.
    im_ndxi[im_ndxi < -1000.0] = -1000
    im_ndxi[im_ndxi > 1000.0] = 1000

    # Apply Validity Mask
    im_ndxi[np.where(input_buffer[1][0] != 0)] = np.nan
    np.nan_to_num(im_ndxi, copy=False, nan=NODATA_INT16)
    im_ndxi = np.int16(im_ndxi)

    return im_ndxi


def std_convoluted(
    im: np.ndarray, kernel_radius: int, min_value: float, max_value: float
) -> np.ndarray:
    """
    Calculate the std of each pixel
    Based on a convolution with a kernel of 1 (size of the kernel given)

    :param np.ndarray im: input image
    :param int kernel_radius: radius of kernel
    :param float min_value: min value of the input image
    :param float max_value: max value of the input image
    :returns: texture image
    """
    im2 = im**2
    kernel = np.ones((2 * kernel_radius + 1, 2 * kernel_radius + 1))
    ns = kernel.size * np.ones(im.shape)

    # Local mean with convolution
    s = scipy.signal.convolve2d(im, kernel, mode="same", boundary="symm")
    # local mean of the squared image with convolution
    s2 = scipy.signal.convolve2d(im2, kernel, mode="same", boundary="symm")

    # Invalid values will be handled later
    np.seterr(divide="ignore", invalid="ignore")
    res = np.sqrt((s2 - s**2 / ns) / ns)  # std calculation

    # Normalization
    res = 1000 * res / (max_value - min_value)

    res = np.where(np.isnan(res), 0, res)

    return res


def texture_task(
    input_buffers: list, input_profiles: list, params: dict
) -> np.ndarray:
    """
    Compute textures

    :param list input_buffers: [im_vhr, valid_stack]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params:
    dictionary of arguments, must contain the keys "nir", "texture_rad",
    "min_value" and "max_value"
    :returns: texture image
    """
    masked_band = np.ma.array(
        input_buffers[0][params["nir"] - 1],
        mask=input_buffers[1] != 0,
    )
    texture = std_convoluted(
        masked_band.astype(float),
        params["texture_rad"],
        params["min_value"],
        params["max_value"],
    )
    texture = np.where(input_buffers[1] != 0, NODATA_INT16, texture)

    return texture
