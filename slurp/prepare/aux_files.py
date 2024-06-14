#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import otbApplication as otb
import time

from slurp.prepare import geometry


def pekel_recovery(file_ref: str, file_out: str, write: bool = False) -> np.ndarray:
    """
    Recover Occurrence Pekel image in uint8

    :param str file_ref: path to the input reference image
    :param str file_out: path for the recovered Pekel image
    :param bool write: write the output image if True, else keep the image in memory
    :returns: Pekel image recovered
    """
    if write:
        print("Recover Occurrence Pekel file to", file_out)
    else:
        print("Recover Occurrence Pekel file")
    pekel_image = geometry.superimpose(
        "/work/datalake/static_aux/MASQUES/PEKEL/data2021/occurrence/occurrence.vrt",
        file_ref,
        file_out,
        otb.ImagePixelType_uint8,
        write
    )

    return pekel_image.transpose(2, 0, 1)[0]


def pekel_month_recovery(file_ref: str, month: int, file_data_out: str, file_mask_out: str, write: bool = False) -> np.ndarray:
    """
    Recover Monthly Recurrence Pekel image.
    monthlyRecurrence and has_observations are signed int8 but coded on int16.

    :param str file_ref: path to the input reference image
    :param int month: number of the month
    :param str file_data_out: path for the recovered monthly recurrence Pekel image
    :param str file_mask_out: path for the recovered has observations Pekel image
    :param bool write: write the output image if True, else keep the image in memory
    :returns: Pekel image recovered
    """
    if write:
        print("Recover Monthly Recurrence Pekel file to", file_data_out)
    else:
        print("Recover Monthly Recurrence Pekel file")

    pekel_image = geometry.superimpose(
        "/work/datalake/static_aux/MASQUES/PEKEL/data2021/MonthlyRecurrence/"
        f"monthlyRecurrence{month}/monthlyRecurrence{month}.vrt",
        file_ref,
        file_data_out,
        otb.ImagePixelType_int16,
        write
    )

    pekel_mask_out = geometry.superimpose(
        "/work/datalake/static_aux/MASQUES/PEKEL/data2021/MonthlyRecurrence/"
        f"has_observations{month}/has_observations{month}.vrt",
        file_ref,
        file_mask_out,
        otb.ImagePixelType_int16,
        write
    )

    return pekel_image.transpose(2, 0, 1)[0]


def hand_recovery(file_ref: str, file_out: str, write: bool = False) -> np.ndarray:
    """
    Recover HAND image

    :param str file_ref: path to the input reference image
    :param str file_out: path for the recovered HAND image
    :param bool write: write the output image if True, else keep the image in memory
    :returns: HAND image recovered
    """
    if write:
        print("Recover HAND file to", file_out)
    else:
        print("Recover HAND file")
    hand_image = geometry.superimpose(
        "/work/datalake/static_aux/MASQUES/HAND_MERIT/" "hnd.vrt",
        file_ref,
        file_out,
        otb.ImagePixelType_float,
        write
    )

    return hand_image.transpose(2, 0, 1)[0]


def cloud_from_gml(file_cloud: str, file_ref: str) -> np.ndarray:
    """
    Compute cloud mask from GML file

    :param str file_cloud: path to the GML file
    :param str file_ref: path to the input reference image
    :returns: cloud mask
    """
    start_time = time.time()
    app = otb.Registry.CreateApplication("Rasterization")
    app.SetParameterString("in", file_cloud)
    app.SetParameterString("im", file_ref)
    app.SetParameterFloat("background", 0)
    app.SetParameterString("mode", "binary")
    app.SetParameterFloat("mode.binary.foreground", 1)
    app.Execute()

    mask_cloud = app.GetImageAsNumpyArray(
        "out", otb.ImagePixelType_uint8
    ).astype(np.uint8)
    print("Rasterize clouds in", time.time() - start_time, "seconds.")

    return mask_cloud
