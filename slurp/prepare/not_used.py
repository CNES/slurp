#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import otbApplication as otb

from slurp.prepare import geometry


def cloud_from_gml(file_cloud: str, file_ref: str) -> np.ndarray:
    """
    Compute cloud mask from GML file

    :param str file_cloud: path to the GML file
    :param str file_ref: path to the input reference image
    :returns: cloud mask
    """
    mask_cloud = geometry.rasterization(
        file_cloud,
        file_ref,
        "",
        otb.ImagePixelType_uint8,
        write=False
    )

    return mask_cloud
