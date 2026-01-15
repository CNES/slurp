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


"""Brings together the auxiliary files reading functions"""

import logging

import numpy as np

from slurp.prepare import geometry

logger = logging.getLogger("slurp")


def aux_file_recovery(
    file_ref: str,
    global_data: str,
    reprojected_data: str,
    grid_sensor: tuple = (),
    grid_geo: tuple = (),
    all_coords: np.ndarray = None,
    roi: np.ndarray = None,
):
    """
    Recover Global Data and reproject it into input reference image geometry (geo or sensor)

    :param str file_ref: path to the input reference image
    :param str global_data: path to the input Pekel global image (tile or .vrt)
    :param str reprojected_data: path for the recovered Pekel image
    :param bool sensor_mode: true if file_ref is in sensor mode (not georeferenced)
    :returns: global data cropped onto target image geometry
    """
    logger.info(
        f"Recover file {global_data=} to {reprojected_data=} onto {file_ref=} geometry"
    )
    geometry.sensor_projection(
        global_data,
        file_ref,
        reprojected_data,
        grid_sensor,
        grid_geo,
        all_coords,
        roi,
    )
