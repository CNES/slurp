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

""" Brings together the geometry functions using OTB features"""

import time
import otbApplication as otb

from slurp.tools.constant import COMPRESSION


def superimpose(file_in: str, file_ref: str, file_out: str, type_out):
    """
    Superimpose using OTB

    :param str file_in: path to the image to reproject into the geometry of the reference input
    :param str file_ref: path to the input reference image
    :param str file_out: path for the output reprojected image
    :param type_out: OTB type for the output image
    """
    start_time = time.time()
    app = otb.Registry.CreateApplication("Superimpose")
    app.SetParameterString("inm", file_in)
    app.SetParameterString("inr", file_ref)
    app.SetParameterString("interpolator", "nn")
    app.SetParameterString("out", file_out + f"?&writerpctags=true&gdal:co:COMPRESS={COMPRESSION}")
    app.SetParameterOutputImagePixelType("out", type_out)
    app.ExecuteAndWriteOutput()

    print("Superimpose in", time.time() - start_time, "seconds.")


def rasterization(file_in: str, file_ref: str, file_out: str, type_out):
    """
    Rasterization using OTB

    :param str file_in: path to the image to rasterize
    :param str file_ref: path to the input reference image
    :param str file_out: path for the output reprojected image
    :param type_out: OTB type for the output image
    """
    start_time = time.time()
    app = otb.Registry.CreateApplication("Rasterization")
    app.SetParameterString("in", file_in)
    app.SetParameterString("im", file_ref)
    app.SetParameterFloat("background", 0)
    app.SetParameterString("mode", "binary")
    app.SetParameterFloat("mode.binary.foreground", 1)
    app.SetParameterString("out", file_out + f"?&writerpctags=true&gdal:co:COMPRESS={COMPRESSION}")
    app.SetParameterOutputImagePixelType("out", type_out)
    app.ExecuteAndWriteOutput()

    print("Rasterize in", time.time() - start_time, "seconds.")
