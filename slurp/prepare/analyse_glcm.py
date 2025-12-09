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

"""
Use a global land cover map to calculate the better number of vegetation cluster
to use for mask computation
"""

import logging

import numpy as np
import rasterio as rio

from slurp.prepare import geometry

logger = logging.getLogger("slurp")


def get_advices(veg, low_veg, high_veg, nb_total):
    """
    Returns adviced number of clusters for vegetation and high vegetation
    regarding ratio of these classes in the image

    :param int veg: number of pixels corresponding to vegetation
    :param int low_veg: number of pixels corresponding to low vegetation
    :param int high_veg: number of pixels corresponding to high vegetation
    :param int nb_total: total number of pixels
    :returns: number of clusters for vegetation and low vegetation
    """
    pct_veg = 100 * veg / nb_total
    if pct_veg == 0:
        return 0, 0
    nb_clusters_veg = 3
    if pct_veg < 5:
        nb_clusters_veg = 1
    elif pct_veg < 25:
        nb_clusters_veg = 2
    elif 60 < pct_veg <= 85:
        nb_clusters_veg = 4
    elif pct_veg > 85:
        nb_clusters_veg = 5

    nb_clusters_low_veg = 3
    pct_low_veg = 100 * low_veg / (low_veg + high_veg)

    if pct_low_veg < 5:
        nb_clusters_low_veg = 1
    elif pct_low_veg < 25:
        nb_clusters_low_veg = 2
    elif 60 < pct_low_veg <= 85:
        nb_clusters_low_veg = 4
    elif pct_low_veg > 85:
        nb_clusters_low_veg = 5

    return nb_clusters_veg, nb_clusters_low_veg


def compute_stats(
    im: str, map_lc: str, cropped: bool, sensor_mode: bool
) -> tuple:
    """
    Compute ratio of vegetation, low vegetation and vegetation in the ROI.
    ESA world cover classification:
        10: "Tree cover",
        20: "Shrubland",
        30: "Grassland",
        40: "Cropland",
        50: "Built-up",
        60: "Bare / Sparse vegetation",
        70: "Snow and ice",
        80: "Permanent water bodies",
        90: "Herbaceous wetland",
        95: "Mangroves",
        100: "Moss and lichen",

    :param str im: path to the input VHR image
    :param str map_lc: path to the land cover map
    :param bool cropped: whether the land cover map only contains the ROI of
                         the input image or is larger
    :returns: number of clusters for vegetation and low vegetation,
              + proportion of pixels of  veg, low_veg, high_veg, non_veg
    """
    if not cropped:
        # get ROI before computing stats
        if sensor_mode:
            logger.error(
                "ERROR : GLCM analysis not implemented for sensor mode yet. "
                "Returns default clustering values"
            )
            return 3, 3
        data_map = geometry.get_extract_roi(map_lc, im)
    else:
        # get all data from map_lc
        ds = rio.open(map_lc)
        data_map = ds.read()
        ds.close()
        del ds

    width = data_map.shape[1]
    height = data_map.shape[2]

    nb_total = width * height
    unique, counts = np.unique(data_map, return_counts=True)

    veg, low_veg, high_veg, non_veg = 0, 0, 0, 0
    vegetation_classes = [10, 20, 30, 90, 95, 100]
    undefined_vegetation_classes = [40]  # crop-lands may be vegetated or not
    low_vegetation_classes = [20, 30, 40, 90, 100]
    high_vegetation_classes = [10, 95]
    non_vegetation_classes = [70, 80]
    logger.debug("Count nb of pixels per class")

    for v, c in zip(unique, counts, strict=False):
        logger.debug(f"{v} : {c}")
        if v in vegetation_classes:
            veg += c

        if v in undefined_vegetation_classes:
            veg += int(0.5 * c)

        if v in low_vegetation_classes:
            low_veg += c

        if v in high_vegetation_classes:
            high_veg += c

        if v in non_vegetation_classes:
            non_veg += c

    nb_clusters_veg, nb_clusters_low_veg = get_advices(
        veg, low_veg, high_veg, nb_total
    )

    veg_ratio = veg / nb_total
    low_veg_ratio = low_veg / nb_total
    high_veg_ratio = high_veg / nb_total
    non_veg_ratio = non_veg / nb_total

    logger.info(f"Vegetation (% area) \t: {100*veg_ratio:.2f}%")
    logger.info(f"Low vegetation (% area) \t: {100*low_veg_ratio:.2f}%")
    logger.info(f"High vegetation (% area) \t: {100*high_veg_ratio:.2f}%")

    logger.info(f"VEG_CLUSTERS : {nb_clusters_veg}")
    logger.info(f"LOW_VEG_CLUSTERS : {nb_clusters_low_veg}")

    return (
        nb_clusters_veg,
        nb_clusters_low_veg,
        veg_ratio,
        low_veg_ratio,
        high_veg_ratio,
        non_veg_ratio,
    )
