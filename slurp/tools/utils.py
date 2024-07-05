#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from skimage.morphology import binary_dilation, disk


def convert_time(seconds):
    full_time = time.gmtime(seconds)
    return time.strftime("%H:%M:%S", full_time)


def compute_mask(im_ref: np.ndarray, thresh_ref: list) -> list:
    """
    Compute mask with one or multiple threshold values

    :param np.ndarray im_ref: input image
    :param list thresh_ref: list of threshold values
    :returns: list of masks for each threshold value
    """
    mask_ref = []
    for thresh in thresh_ref:
        mask_ref.append(im_ref > thresh)

    return mask_ref


def compute_mask_threshold(input_buffers: list, input_profiles: list, params: dict) -> np.ndarray:
    """
    Compute boolean mask with threshold value

    :param list input_buffers: Input image and valid stack [input_image, valid_stack]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments, must contain the key "threshold"
    :returns: computed mask
    """
    mask = np.where(input_buffers[0][0] > params["threshold"], 1, 0)
    mask = np.where(input_buffers[1][0] != 1, 255, mask)

    return mask


def concatenate_samples(output_scalars, chunk_output_scalars, tile):
    output_scalars.append(chunk_output_scalars[0])
