#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Brings together the morphology functions """

import numpy as np
from skimage.morphology import (area_closing, binary_closing, binary_dilation, binary_erosion, binary_opening,
                                remove_small_holes, remove_small_objects, disk)


def apply_morpho(input_array: np.ndarray, key: str, value: int) -> np.ndarray:
    """
    Apply the selected morphology transformation
    
    :params array input_array: 
    :params str key: Name of the morpho to apply
    :params int value: (depend of the key selected)
    """
    
    if key == "area_closing":
        output_array = area_closing(input_array, value, connectivity=2)
    elif key == "binary_closing":
        output_array = binary_closing(input_array, disk(value))
    elif key == "binary_dilation":
        output_array = binary_dilation(input_array, disk(value))
    elif key == "binary_erosion":
        output_array = binary_erosion(input_array, disk(value))
    elif key == "binary_opening":
        output_array = binary_opening(input_array, disk(value))
    elif key == "remove_small_holes":
        output_array = remove_small_holes(input_array, value, connectivity=2)
    elif key == "remove_small_objects":
        output_array = remove_small_objects(input_array, value, connectivity=2)
    else:
        raise NotImplementedError(f"The key {key} is not implemented")

    return output_array


def morpho_clean(im_classif, params):
    """
    Apply the morphology transformation passed in arguments
    
    :params array im_classif: 
    :param dict params: dictionary of arguments
    """
    
    im_classif = im_classif.astype(np.uint8)

    if params["binary_closing"]:
        # Closing can remove small dark spots (i.e. “pepper”) and connect small bright cracks.
        im_classif = apply_morpho(im_classif, "binary_closing", params["binary_closing"]).astype(np.uint8)

    if params["binary_opening"]:
        # Opening can remove small bright spots (i.e. “salt”) and connect small dark cracks.
        im_classif = apply_morpho(im_classif, "binary_opening", params["binary_opening"]).astype(np.uint8)

    if params["remove_small_holes"]:
        im_classif = apply_morpho(
            im_classif.astype(bool), "remove_small_holes", params["remove_small_holes"]
        ).astype(np.uint8)

    if params["remove_small_objects"]:
        im_classif = apply_morpho(
            im_classif.astype(bool), "remove_small_objects", params["remove_small_objects"]
        ).astype(np.uint8)

    return im_classif
