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


"""Brings together functions that create valid mask"""

import numpy as np


def compute_valid_stack_clouds(
    input_buffers: list, input_profiles: list, args: dict
) -> np.ndarray:
    """
    Calculation of the valid pixels of a given image with a cloud mask

    :param list input_buffer: VHR input image [im_vhr, mask_cloud]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict args: dictionary of arguments, must contain a key "nodata"
    :returns: valid_mask (numpy array, 0 : valid, 1 : NODATA, 2 : Clouds)
    """
    if len(input_buffers) == 1:
        # 0 where image is valid, invalid for other values
        # Note : we check no data on all dimensions, because in some weird cases
        # (ex : RGB bands and NIR had been superimposed on a slightly different grid)
        # some pixels may be valid in band 0 (Red)  and not in band 3 (NIR)
        valid_mask = 1 - np.all(
            input_buffers[0] != args["nodata"], axis=0
        ).astype(int)
    else:
        valid_mask = 1 - np.all(
            input_buffers[0] != args["nodata"], axis=0
        ).astype(int)
        valid_mask = np.where(input_buffers[1] != 0, 2, valid_mask)

    return valid_mask
