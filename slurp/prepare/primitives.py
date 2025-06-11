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
    im_ndxi[np.logical_or(im_ndxi < -1000.0, im_ndxi > 1000.0)] = np.nan
    im_ndxi[np.logical_not(input_buffer[1][0])] = np.nan
    np.nan_to_num(im_ndxi, copy=False, nan=NODATA_INT16)
    im_ndxi = np.int16(im_ndxi)

    return im_ndxi
