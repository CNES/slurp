#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from skimage.morphology import binary_dilation, disk


def compute_valid_stack(input_buffer: list, input_profiles: list, args: dict) -> np.ndarray:
    """
    Calculation of the valid pixels of a given image

    :param list input_buffer: VHR input image [im_vhr]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict args: dictionary of arguments, must contain a key "nodata"
    :returns: valid_phr (boolean numpy array, True = valid data, False = no data)
    """
    valid_phr = np.logical_and.reduce(input_buffer[0] != args["nodata"], axis=0)
    return valid_phr


def compute_valid_stack_clouds(input_buffer: list, input_profiles: list, args: dict) -> np.ndarray:
    """
    Calculation of the valid pixels of a given image with a cloud mask

    :param list input_buffer: VHR input image [im_vhr, mask_nocloud]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict args: dictionary of arguments, must contain a key "nodata"
    :returns: valid_phr (boolean numpy array, True = valid data, False = no data)
    """
    valid_phr = np.logical_and.reduce(input_buffer[0] != args["nodata_phr"], axis=0)
    valid_stack_cloud = np.logical_and(valid_phr, input_buffer[1])

    return valid_stack_cloud


def compute_valid_stack_masks(input_buffer: list, input_profiles: list, args: dict) -> np.ndarray:
    """
    Calculation of the valid pixels of a given image with a cloud mask and vegetation or water mask

    :param list input_buffer: VHR input image [im_vhr, mask_nocloud, vegetationmask, watermask]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict args: dictionary of arguments, must contain the keys "nodata", "vegetationmask" and "watermask"
    :returns: valid_phr (boolean numpy array, True = valid data, False = no data)
    """
    valid_phr = np.logical_and.reduce(input_buffer[0] != args["nodata_phr"], axis=0)
    valid_stack = np.logical_and(valid_phr, input_buffer[1])

    if args["vegetationmask"] is not None:
        non_veg = np.where(input_buffer[2] < args["vegmask_max_value"], True, False)
        # dilate non vegetation areas, because sometimes the vegetation mask can cover urban areas
        non_veg_dilated = binary_dilation(non_veg[0], disk(args["binary_dilation"]))
        valid_stack = np.logical_and(valid_stack, [non_veg_dilated])

    if args["watermask"] is not None:
        valid_stack = np.logical_and(valid_stack, np.where(input_buffer[3] == 0, True, False))

    return valid_stack


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
