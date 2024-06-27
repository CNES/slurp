#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def compute_valid_stack(input_buffer: list, input_profiles: list, args: dict) -> np.ndarray:
    """
    Calculation of the valid pixels of a given image

    :param list input_buffer: VHR input image [im_vhr]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict args: dictionary of arguments, must contain a key "nodata"
    :returns: valid_mask (boolean numpy array, True = valid data, False = no data)
    """
    valid_mask = np.logical_and.reduce(input_buffer[0] != args["nodata"], axis=0)
    return valid_mask


def compute_valid_stack_clouds(input_buffer: list, input_profiles: list, args: dict) -> np.ndarray:
    """
    Calculation of the valid pixels of a given image with a cloud mask

    :param list input_buffer: VHR input image [im_vhr, mask_cloud]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict args: dictionary of arguments, must contain a key "nodata"
    :returns: valid_mask (boolean numpy array, True = valid data, False = no data)
    """
    valid_phr = np.logical_and.reduce(input_buffer[0] != args["nodata_phr"], axis=0)
    no_cloud = (input_buffer[1] == 0)
    valid_mask = np.logical_and(valid_phr, no_cloud)

    return valid_mask
