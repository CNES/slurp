#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def compute_ndvi(input_buffer: list, input_profiles: list, params: dict) -> np.ndarray:
    """
    Compute Normalize Difference X Index.
    Rescale to [-1000, 1000] int16 with nodata value = 32767
    1000 * (im_b1 - im_b2) / (im_b1 + im_b2)

    :param list input_buffer: VHR input image [im_vhr] or [im_vhr, valid_stack]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments, must contain the keys "red" and "nir"
    :returns: NDVI
    """
    np.seterr(divide="ignore", invalid="ignore")
    im_ndvi = 1000.0 - (2000.0 * np.float32(input_buffer[0][params["red"] - 1])) / (
            np.float32(input_buffer[0][params["nir"] - 1]) + np.float32(input_buffer[0][params["red"] - 1]))
    im_ndvi[np.logical_or(im_ndvi < -1000.0, im_ndvi > 1000.0)] = np.nan
    if len(input_buffer) > 1:  # for urbanmask or vegmask computation
        im_ndvi[np.logical_not(input_buffer[1][0])] = np.nan
    np.nan_to_num(im_ndvi, copy=False, nan=32767)
    im_ndvi = np.int16(im_ndvi)

    return im_ndvi


def compute_ndwi(input_buffer: list, input_profiles: list,  params: dict) -> np.ndarray:
    """
    Compute Normalize Difference X Index.
    Rescale to [-1000, 1000] int16 with nodata value = 32767
    1000 * (im_b1 - im_b2) / (im_b1 + im_b2)

    :param list input_buffer: VHR input image [im_vhr] or [im_vhr, valid_stack]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments, must contain the keys "green" and "nir"
    :returns: NDWI
    """
    np.seterr(divide="ignore", invalid="ignore")
    im_ndwi = 1000.0 - (2000.0 * np.float32(input_buffer[0][params["nir"] - 1])) / (
        np.float32(input_buffer[0][params["green"] - 1]) + np.float32(input_buffer[0][params["nir"] - 1]))
    im_ndwi[np.logical_or(im_ndwi < -1000.0, im_ndwi > 1000.0)] = np.nan
    if len(input_buffer) > 1:  # for urbanmask or vegmask computation
        im_ndwi[np.logical_not(input_buffer[1][0])] = np.nan
    np.nan_to_num(im_ndwi, copy=False, nan=32767)
    im_ndwi = np.int16(im_ndwi)

    return im_ndwi
