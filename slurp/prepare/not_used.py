#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import otbApplication as otb

from slurp.prepare import geometry


def cloud_from_gml(file_cloud: str, file_ref: str):
    """
    Compute cloud mask from GML file

    :param str file_cloud: path to the GML file
    :param str file_ref: path to the input reference image
    """
    geometry.rasterization(
        file_cloud,
        file_ref,
        "",
        otb.ImagePixelType_uint8
    )
