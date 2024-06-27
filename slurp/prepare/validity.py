#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def compute_valid_stack(input_buffer: list, input_profiles: list, args: dict) -> np.ndarray:
    """
    Calculation of the valid pixels of a given image

    :param list input_buffer: VHR input image [im_vhr]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict args: dictionary of arguments, must contain a key "nodata"
    :returns: valid_phr (boolean numpy array, True = valid data, False = no data)
    """
    valid_mask = np.logical_and.reduce(input_buffer[0] != args["nodata"], axis=0)
    return valid_mask
