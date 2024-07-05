#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import otbApplication as otb
import time


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
    app.SetParameterString("out", file_out + "?&writerpctags=true&gdal:co:COMPRESS=DEFLATE")
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
    app.SetParameterString("out", file_out + "?&writerpctags=true&gdal:co:COMPRESS=DEFLATE")
    app.SetParameterOutputImagePixelType("out", type_out)
    app.ExecuteAndWriteOutput()

    print("Rasterize in", time.time() - start_time, "seconds.")
