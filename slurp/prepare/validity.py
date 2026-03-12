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
from typing import Optional


def compute_valid_stack_clouds(
    im_vhr: np.ndarray,
    mask_cloud: Optional[np.ndarray] = None,
    nodata: float = None,
) -> np.ndarray:
    """
    Calculation of valid pixels of a given image with optional cloud mask.

    Parameters
    ----------
    im_vhr : np.ndarray
        Input VHR image tile (C, H, W) or (H, W)
    mask_cloud : np.ndarray, optional
        Cloud mask tile
    nodata : numeric
        Nodata value

    Returns
    -------
    np.ndarray
        valid_mask:
            0 ? valid
            1 ? nodata
            2 ? cloud
    """
    # check NODATA on all bands : in very rare cases, we observe images
    # with invalid data on only one band
    valid_mask = 1 - np.all(im_vhr != nodata, axis=0).astype(int)
    if mask_cloud is not None:
        valid_mask = np.where(mask_cloud != 0, 2, valid_mask)

    return valid_mask.astype(np.uint8)
