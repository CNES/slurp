#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import otbApplication as otb
import scipy
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
    mask_cloud = geometry.rasterization(
        file_cloud,
        file_ref,
        "",
        otb.ImagePixelType_uint8,
        write=False
    )

    return mask_cloud


def wsf_recovery(file_ref: str, file_out: str, write=False) -> np.ndarray:
    """
    Recover WSF image in uint16

    :param str file_ref: path to the input reference image
    :param str file_out: path for the recovered WSF image
    :param bool write: write the output image if True, else keep the image in memory
    :returns: WSF image recovered
    """
    if write:
        print("Recover WSF file to", file_out)
    else:
        print("Recover WSF file")
    wsf_image = geometry.superimpose(
        "/work/datalake/static_aux/MASQUES/WSF/WSF2019_v1/WSF2019_v1.vrt",
        file_ref,
        file_out,
        otb.ImagePixelType_uint16,
        write
    )

    return wsf_image.transpose(2, 0, 1)[0]


def std_convoluted(im: np.ndarray, N: int, min_value: float, max_value: float) -> np.ndarray:
    """
    Calculate the std of each pixel
    Based on a convolution with a kernel of 1 (size of the kernel given)

    :param np.ndarray im: input image
    :param int N: radius of kernel
    :param float min_value: min value of the input image
    :param float max_value: max value of the input image
    :returns: texture image
    """
    im2 = im ** 2
    kernel = np.ones((2 * N + 1, 2 * N + 1))
    ns = kernel.size * np.ones(im.shape)

    # Local mean with convolution
    s = scipy.signal.convolve2d(im, kernel, mode="same", boundary="symm")
    # local mean of the squared image with convolution
    s2 = scipy.signal.convolve2d(im2, kernel, mode="same", boundary="symm")

    # Invalid values will be handled later
    np.seterr(divide="ignore", invalid="ignore")
    res = np.sqrt((s2 - s ** 2 / ns) / ns)  # std calculation

    # Normalization
    res = 1000 * res / (max_value - min_value)

    res = np.where(np.isnan(res), 0, res)

    return res


def texture_task(input_buffers: list, input_profiles: list, params: dict) -> np.ndarray:
    """
    Compute textures

    :param list input_buffers: [im_vhr, valid_stack]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments, must contain the keys "nir", "texture_rad", "min_value" and "max_value"
    :returns: texture image
    """
    masked_band = np.ma.array(input_buffers[0][params["nir"] - 1], mask=np.logical_not(input_buffers[1]))
    texture = std_convoluted(masked_band.astype(float), params["texture_rad"], params["min_value"], params["max_value"])

    return texture
