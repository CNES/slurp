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
    input_buffer_b1: np.ndarray,
    input_buffer_b2: np.ndarray,
    valid_stack: np.ndarray,
    im_b1: int,
    im_b2: int,
) -> np.ndarray:
    """
    Compute Normalized Difference X Index (NDXI).

    Formula:
        1000 * (B1 - B2) / (B1 + B2)

    The result is clipped to [-1000, 1000] and converted to int16.
    Nodata value is set to NODATA_INT16 (32767).

    Special case:
    If reflectance values are negative, the index may slightly exceed
    the theoretical range [-1000, 1000]. In that case, values are clipped
    to -1000 or 1000.

    :param np.ndarray input_buffer: Multi-band image array
        Shape expected: (bands, height, width)
    :param np.ndarray valid_stack: Validity mask
        Non-zero values indicate invalid pixels
    :param int im_b1: 1-based index of band B1
    :param int im_b2: 1-based index of band B2
    :returns: NDXI image as np.int16 array
    :rtype: np.ndarray
    """
    # Convert selected bands to float32 for safe division
    band_b1 = input_buffer_b1.astype(np.float32)
    band_b2 = input_buffer_b2.astype(np.float32)

    denom = band_b1 + band_b2

    with np.errstate(divide="ignore", invalid="ignore"):
        im_ndxi = 1000.0 - (2000.0 * band_b2) / denom

    # Clip to theoretical range
    np.clip(im_ndxi, -1000.0, 1000.0, out=im_ndxi)

    # Apply validity mask (non-zero = invalid)
    im_ndxi[valid_stack != 0] = np.nan

    # Replace NaN with NODATA value
    np.nan_to_num(im_ndxi, copy=False, nan=NODATA_INT16)
    return im_ndxi.astype(np.int16)


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
    nir_band: np.ndarray,
    valid_stack: np.ndarray,
    nir: int,
    texture_rad: int,
    min_value: float,
    max_value: float,
) -> np.ndarray:
    """
    Compute texture on NIR band.

    :param np.ndarray nir_band:
        nir-band image array (height, width)
    :param np.ndarray valid_stack:
        Validity mask (non-zero = invalid)
    :param int nir:
        1-based index of NIR band
    :param int texture_rad:
        Radius of texture window
    :param float min_value:
        Lower percentile clipping value
    :param float max_value:
        Upper percentile clipping value
    :returns:
        Texture image as np.uint16
    :rtype: np.ndarray
    """

    # Select NIR band (convert to float)
    masked_band = np.ma.array(
        nir_band.astype(float),
        mask=valid_stack != 0,
    )

    texture = std_convoluted(
        masked_band,
        texture_rad,
        min_value,
        max_value,
    )

    texture = np.where(valid_stack != 0, NODATA_INT16, texture)

    return texture.astype(np.uint16)
