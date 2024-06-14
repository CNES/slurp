#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import otbApplication as otb
import time


def superimpose(file_in: str, file_ref: str, file_out: str, type_out, write: bool = False) -> np.ndarray:
    """
    Superimpose using OTB

    :param str file_in: path to the image to reproject into the geometry of the reference input
    :param str file_ref: path to the input reference image
    :param str file_out: path for the output reprojected image
    :param type_out: OTB type for the output image
    :param bool write: write the output image if True, else keep the image in memory
    :returns: reprojected image
    """
    start_time = time.time()
    app = otb.Registry.CreateApplication("Superimpose")
    app.SetParameterString("inm", file_in)
    app.SetParameterString("inr", file_ref)
    app.SetParameterString("interpolator", "nn")
    app.SetParameterString("out", file_out + "?&writerpctags=true")
    app.SetParameterOutputImagePixelType("out", type_out)
    app.Execute()

    res = np.int16(np.copy(app.GetVectorImageAsNumpyArray("out")))

    if write:
        app.WriteOutput()

    print("Superimpose in", time.time() - start_time, "seconds.")

    return res
