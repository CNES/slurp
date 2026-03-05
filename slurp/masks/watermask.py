#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of SLURP
# (see https://github.com/CNES/slurp).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Compute water mask of PHR image with help of Pekel and Hand images."""

import argparse
import gc
import logging
import time
import traceback
from os import makedirs, path
from copy import deepcopy

import eoscale.eo_executors as eoexe
import eoscale.manager as eom
import numpy as np
from skimage.measure import label, regionprops
from sklearn.ensemble import RandomForestClassifier

from slurp.eomultiprocessing.slurp_executor import mp_n_to_m_images, mp_n_to_m_scalars
from slurp.eomultiprocessing.slurp_manager import slurpContextManager
from slurp.eomultiprocessing.utils import read_and_get_profile, write, read
from slurp.post_process.morphology import apply_morpho
from slurp.tools import profile_utils as eo_utils
from slurp.tools import random_forest_utils as rf_utils
from slurp.tools import utils
from slurp.tools.constant import NODATA_INT8

logger = logging.getLogger("slurp")

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ModuleNotFoundError:
    logger.error("Intel(R) Extension/Optimization for scikit-learn not found.")


def compute_pekel_mask(
    pekel_img: np.ndarray,
    thresh_pekel: float,
    hand_strict: bool,
    strict_thresh: float,
    no_pekel_filter: bool,
    pekel_nodata: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Pekel mask regarding entry arguments.

    Parameters
    ----------
    input_buffer : np.ndarray
        Pekel image
    thresh_pekel : float
    hand_strict : bool
    strict_thresh : float
    no_pekel_filter : bool
    pekel_nodata : float

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (mask_pekel, secondary_mask)
    """
    # Main water mask
    mask_pekel = utils.compute_mask(pekel_img, [thresh_pekel])

    if hand_strict:
        mask_pekel_strict = utils.compute_mask(
            pekel_img, [strict_thresh]
        )
        return mask_pekel, mask_pekel_strict

    if not no_pekel_filter:
        mask_pekel0 = utils.compute_mask(pekel_img, [0])
    else:
        mask_pekel0 = np.zeros(pekel_img.shape, dtype=np.uint8)

    return mask_pekel, mask_pekel0


def compute_hand_mask(
    hand_array: np.ndarray, thresh_hand: int,
) -> bool:
    """
    Compute Hand mask with one or multiple threshold values.

    :param list input_buffer: Hand image [hand_image]
    :param dict params: dictionary of arguments
    :returns: Hand mask (true if pixels are below a "thresh_hand" altitude)
    """
    mask_hand = hand_array > thresh_hand

    # Do not learn in water surface (useful if image contains big water surfaces)
    # Add some robustness if hand_strict is not used
    # if args.hand_strict:
    # np.logical_not(np.logical_or(mask_hand, inputBuffer[1]), out=mask_hand)
    # else:
    # np.logical_not(mask_hand, out=mask_hand)
    mask_hand = np.logical_not(mask_hand, out=mask_hand)

    return mask_hand


def get_random_indexes_from_masks(
    nb_samples: int, mask_1: np.ndarray, mask_2: np.ndarray
):
    """
    Get random valid indexes from masks.
    :param int nb_samples : number of indices to count
    :param np.ndarray Mask 1 is a validity mask - shape (height, width)
    :param np.ndarray Mask 2 is the reference data mask - shape (height, width)
    """
    np.random.seed(712)  # reproductible results
    rows_idxs = []
    cols_idxs = []

    if nb_samples != 0:
        nb_idxs = 0

        height = mask_1.shape[0]
        width = mask_1.shape[1]

        while nb_idxs < nb_samples:
            row = np.random.randint(0, height)
            col = np.random.randint(0, width)
            if mask_1[row, col] and mask_2[row, col]:
                rows_idxs.append(row)
                cols_idxs.append(col)
                nb_idxs += 1

    return rows_idxs, cols_idxs


def get_grid_indexes_from_mask(
    nb_samples: int, valid_mask: np.ndarray, mask_ground_truth: np.ndarray
):
    """
    Retrieve row and columns indices selected on the valid pixel of the image

    :param int nb_samples: number of samples selected
    :param boolean numpy array valid_mask : shape (height, width)
    :param boolean numpy array mask_ground_truth :  shape (height, width)
    :return: tuple of list , row indices and columns indices
    """
    valid_samples = np.logical_and(mask_ground_truth, valid_mask).astype(
        np.uint8
    )
    rows, cols = np.where(valid_samples)

    if 1 <= nb_samples <= len(rows):
        # np.arange(0, len(rows) -1, ...) : to be sure to exclude index len(rows)
        # because in some cases (ex : 19871, 104 samples), last index is the len(rows)
        indices = np.arange(0, len(rows) - 1, len(rows) / nb_samples).astype(
            np.uint16
        )

        s_rows = rows[indices]
        s_cols = cols[indices]
    else:
        s_rows = []
        s_cols = []

    return s_rows, s_cols


def get_smart_indexes_from_mask(nb_samples, pct_area, minimum, mask):
    """
    Retrieve row and columns indices selected on the valid pixel of the image

    :param int nb_samples: number of samples selected
    :param int pct_area: importance of area for selecting number of samples in each water surface
    :param int minimum: minimum number of samples in each water surface
    :param boolean numpy array mask: validity mask (shape (height, width))
    :return: tuple of list , row indices and columns indices

    """
    rows_idxs = []
    cols_idxs = []

    if nb_samples != 0:
        img_labels, nb_labels = label(mask, return_num=True)
        props = regionprops(img_labels)
        mask_area = float(np.sum(mask))

        # number of samples for each label/prop
        n1_indexes = int((1.0 - pct_area / 100.0) * nb_samples / nb_labels)

        # number of samples to distribute to each label/prop
        n2_indexes = pct_area / 100.0 * nb_samples / mask_area

        for prop in props:
            n3_indexes = n1_indexes + int(n2_indexes * prop.area)
            n3_indexes = max(minimum, n3_indexes)

            min_row = np.min(prop.bbox[0])
            max_row = np.max(prop.bbox[2])
            min_col = np.min(prop.bbox[1])
            max_col = np.max(prop.bbox[3])

            nb_idxs = 0
            while nb_idxs < n3_indexes:
                np.random.seed(712)  # reproductible results
                row = np.random.randint(min_row, max_row)
                col = np.random.randint(min_col, max_col)

                if mask[row, col]:
                    rows_idxs.append(row)
                    cols_idxs.append(col)
                    nb_idxs += 1

    return rows_idxs, cols_idxs


def build_samples(
    valid_stack: np.ndarray,
    mask_hand: np.ndarray,
    mask_pekel: np.ndarray,
    phr1: np.ndarray,
    phr2: np.ndarray,
    phr3: np.ndarray,
    phr4: np.ndarray,
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    ndwi_threshold:float,
    nb_valid_water_pixels: int,
    nb_valid_other_pixels: int,
    nb_samples_water: int,
    nb_samples_other: int,
    samples_method: str,
    nb_samples_auto: bool,
    auto_pct: float,
    smart_area_pct: float,
    smart_minimum: int,
) -> np.ndarray:
    """
    Build samples from tiled ndarray inputs (SLURP executor version).

    Each argument is a tile extracted by mp_n_to_m_scalars.
    """

    # ---- validity mask ----
    validity_mask = (valid_stack == 0)
    # ---- valid water pixels ----
    valid_water_pixels = np.logical_and(
        mask_pekel,
        ndwi > ndwi_threshold,
    )
    valid_water_pixels = np.logical_and(valid_water_pixels, validity_mask)

    nb_water_subset = np.count_nonzero(valid_water_pixels)

    # ---- valid "other" pixels ----
    nb_valid_pix_other = np.count_nonzero(
        np.logical_and(validity_mask, mask_hand)
    )
    nb_other_subset = nb_valid_pix_other

    logger.debug(
        f"DBG> {nb_water_subset=} {nb_other_subset=}"
    )

    # ---- ratios ----
    water_ratio = nb_water_subset / nb_valid_water_pixels
    other_ratio = nb_other_subset / nb_valid_other_pixels

    nb_water_subsamples = round(water_ratio * nb_samples_water)
    nb_other_subsamples = round(other_ratio * nb_samples_other)

    # ---- AUTO MODE ----
    if nb_samples_auto:
        nb_water_subsamples = int(nb_water_subset * auto_pct)
        nb_other_subsamples = int(nb_other_subset * auto_pct)
    
    # ---- WATER SAMPLES ----
    if samples_method == "random":

        rows_pekel, cols_pekel = get_random_indexes_from_masks(
            nb_water_subsamples,
            validity_mask,
            mask_pekel,
        )

    elif samples_method == "smart":

        rows_pekel, cols_pekel = get_smart_indexes_from_mask(
            nb_water_subsamples,
            smart_area_pct,
            smart_minimum,
            np.logical_and(mask_pekel, validity_mask),
        )

    elif samples_method == "grid":

        rows_pekel, cols_pekel = get_grid_indexes_from_mask(
            nb_water_subsamples,
            validity_mask,
            valid_water_pixels,
        )

    else:
        raise ValueError(
            "samples_method must be 'random', 'smart' or 'grid'"
        )

    # ---- OTHER SAMPLES (always random) ----
    rows_hand, cols_hand = get_random_indexes_from_masks(
        nb_other_subsamples,
        validity_mask,
        mask_hand,
    )

    # ---- merge samples ----
    rows = np.concatenate((rows_pekel, rows_hand))
    cols = np.concatenate((cols_pekel, cols_hand))
    # ---- stack features ----
    im_stack = np.stack(
        (mask_pekel, phr1, phr2, phr3, phr4, ndvi, ndwi),
        axis=0,
    )
    samples = np.transpose(
        im_stack[:, rows.astype(np.uint16), cols.astype(np.uint16)]
    )
    return samples

def rf_prediction(
    valid_stack,
    phr,
    ndvi,
    ndwi,
    *extra_layers,
    classifier,
    debug=False,
):
    """
    Random Forest prediction

    Parameters
    ----------
    valid_stack : np.ndarray
        Validity mask (1, H, W)
    phr : np.ndarray
        PHR features
    ndvi : np.ndarray
        NDVI layer(s)
    ndwi : np.ndarray
        NDWI layer(s)
    *extra_layers : np.ndarray
        Additional feature layers
    classifier : sklearn-like estimator
        Trained classifier
    debug : bool
        Enable memory debug logs

    Returns
    -------
    np.ndarray
        Predicted mask (uint8)
    """

    # ---- build feature stack ----------------------------------------------
    features = (phr, ndvi, ndwi, *extra_layers)
    im_stack = np.stack(features, axis=0)
    # valid_stack shape expected: (1, H, W)
    valid_mask = np.logical_not(valid_stack)

    # ---- reshape for sklearn ----------------------------------------------
    buffer_to_predict = np.transpose(im_stack[:, valid_mask])

    prediction = np.zeros(im_stack.shape[1:], dtype=np.uint8)

    if buffer_to_predict.shape[0] > 0:
        prediction[valid_mask] = classifier.predict(buffer_to_predict)

    # ---- debug -------------------------------------------------------------
    utils.display_mem_usage(
        debug,
        f"RF Prediction on buffer "
        f"{im_stack.shape[1]} x {im_stack.shape[2]}",
    )

    return prediction


def mask_filter(im_in, mask_ref):
    """
    Remove water areas in im_in not in contact
    with water areas in mask_ref.
    """
    im_label, _ = label(im_in, connectivity=2, return_num=True)

    im_label_thresh = np.copy(im_label)
    im_label_thresh[np.logical_not(mask_ref)] = 0
    valid_labels = np.delete(np.unique(im_label_thresh), 0)

    im_filtered = np.zeros(np.shape(mask_ref), dtype=np.uint8)
    im_filtered[np.isin(im_label, valid_labels)] = 1

    return im_filtered


def apply_ndwi_thresh(args, eoscale_manager, key_ndwi, key_valid_stack):
    logger.info("Simple threshold mask NDWI > " + str(args.ndwi_threshold))
    key_predict = eoexe.n_images_to_m_images_filter(
        inputs=[key_ndwi, key_valid_stack],
        image_filter=utils.compute_mask_threshold,
        filter_parameters={"threshold": 1000 * args.ndwi_threshold},
        context_manager=eoscale_manager,
        generate_output_profiles=eo_utils.single_uint8_profile,
        multiproc_context=args.multiproc_context,
        filter_desc="Simple NDWI threshold",
    )
    time_random_forest = time.time()
    time_samples = time_random_forest
    do_post_process = False
    return do_post_process, key_predict, time_random_forest, time_samples


def post_process(
    im_predict,
    mask_hand,
    mask_pekel0,
    valid_stack,
    hand_filter,
    hand_strict,
    no_pekel_filter,
    binary_closing,
    binary_opening,
    area_closing,
    remove_small_holes,
    remove_small_objects,
    value_classif,
) -> tuple:
    """
    Compute filters on the prediction image.

    Parameters
    ----------
    im_predict : np.ndarray
        Predicted mask
    mask_hand : np.ndarray
        Hand mask
    mask_pekel0 : np.ndarray
        Pekel mask
    valid_stack : np.ndarray
        Validity mask

    Keyword-only arguments
    ---------------------
    hand_filter : bool
    hand_strict : bool
    no_pekel_filter : bool
    binary_closing : int
    binary_opening : int
    area_closing : int
    remove_small_holes : int
    remove_small_objects : int
    value_classif : int

    Returns
    -------
    list[np.ndarray]
        [im_predict, im_classif]
    """

    buffer_shape = im_predict.shape

    # ---- Filter with Hand ----
    if hand_filter:
        if not hand_strict:
            im_predict[np.logical_not(mask_hand)] = 0
        else:
            logger.warning(
                "\nWARNING: hand_filter and hand_strict are incompatible."
            )

    # ---- Filter for final classification ----
    if not no_pekel_filter:
        mask = np.zeros(buffer_shape, dtype=bool)
        mask = np.logical_or(mask, mask_pekel0)
        im_classif = mask_filter(im_predict, mask)
    else:
        im_classif = im_predict.copy()

    # ---- Morphological operations ----
    if binary_closing:
        im_classif[:, :] = apply_morpho(
            im_classif[  :, :].astype(bool), "binary_closing", binary_closing
        ).astype(np.uint8)

    if binary_opening:
        im_classif[:, :] = apply_morpho(
            im_classif[:, :].astype(bool), "binary_opening", binary_opening
        ).astype(np.uint8)

    if area_closing:
        im_classif[:, :] = apply_morpho(
            im_classif[:, :], "area_closing", area_closing
        )

    if remove_small_holes:
        im_classif[:, :] = apply_morpho(
            im_classif[:, :].astype(bool), "remove_small_holes", remove_small_holes
        ).astype(np.uint8)

    if remove_small_objects:
        im_classif[:, :] = apply_morpho(
            im_classif[:, :].astype(bool), "remove_small_objects", remove_small_objects
        ).astype(np.uint8)

    # ---- Add nodata ----
    im_classif[valid_stack != 0] = NODATA_INT8
    im_classif[im_classif == 1] = value_classif

    im_predict[valid_stack != 0] = NODATA_INT8
    im_predict[im_predict == 1] = value_classif

    return im_predict, im_classif


def build_stack_water(args, slurp_manager):
    """
    Prepares and returns the required image layers and masks
    for water mask processing using slurpContextManager.

    Parameters
    ----------
    args : Namespace
        Object containing paths and configuration parameters.
        Expected attributes:
            - file_vhr : str
            - valid_stack : str
            - file_ndvi : str
            - file_ndwi : str
        The following attributes will be updated in-place:
            - nodata_phr
            - shape
            - crs
            - transform
            - rpc (set to None)

    slurp_manager : slurpContextManager
        SLURP context manager handling shared memory and raster access.

    Returns
    -------
    key_ndvi : list
    key_ndwi : list
    key_phr : list
    key_valid_stack : list
    margin : int
    """

    # ==============================
    # PHR (VHR image)
    # ==============================

    key_phr, profile_phr = read_and_get_profile(args.file_vhr)

    args.nodata_phr = profile_phr.get("nodata")
    args.shape = (profile_phr["height"], profile_phr["width"])
    args.crs = profile_phr["crs"]
    args.transform = profile_phr["transform"]
    args.rpc = None  # Not handled in slurp mode

    # ==============================
    # VALID STACK
    # ==============================

    key_valid_stack = read(args.valid_stack)

    # ==============================
    # MARGIN (for Pekel projection compatibility)
    # ==============================

    # A Pekel pixel is 30m wide (~45m diagonal).
    # For 0.5m imagery ? 100 pixels safely cover one Pekel pixel.
    margin = 100

    # ==============================
    # NDVI / NDWI
    # ==============================

    key_ndvi = read(args.file_ndvi)
    key_ndwi = read(args.file_ndwi)

    # Wrap into lists for consistency with mp_n_to_m_images API
    return (
        [key_ndvi],
        [key_ndwi],
        [key_phr],
        [key_valid_stack], 
        margin,
        profile_phr
    )


def display_global_infos(args, end_time, t0, time_stack):
    """
    Displays general information about the water mask processing.
    """
    logger.info(
        f"**** Water mask for {args.file_vhr} (saved as {args.watermask}) ****"
    )
    logger.info(
        "Total time (user)       :\t" + utils.convert_time(end_time - t0)
    )
    logger.info(
        "- Build_stack           :\t" + utils.convert_time(time_stack - t0)
    )


def display_rf_infos(end_time, time_random_forest, time_samples, time_stack):
    """
    Displays information about the random forest training and prediction process.
    """
    logger.info(
        "- Build_samples         :\t"
        + utils.convert_time(time_samples - time_stack)
    )
    logger.info(
        "- Random forest (total) :\t"
        + utils.convert_time(time_random_forest - time_samples)
    )
    logger.info(
        "- Post-processing       :\t"
        + utils.convert_time(end_time - time_random_forest)
    )


def display_computation_info(
    args,
    end_time,
    not_enough_water_samples,
    t0,
    time_random_forest,
    time_samples,
    time_stack,
):
    """
    Displays information about the entire computation process, including when handling edge cases.
    """
    logger.info(
        f"**** Water mask for {args.file_vhr} (saved as {args.watermask}) ****"
    )
    logger.info(
        "Total time (user)       :\t" + utils.convert_time(end_time - t0)
    )
    logger.info(
        "- Build_stack           :\t" + utils.convert_time(time_stack - t0)
    )
    if not args.simple_ndwi_threshold and not not_enough_water_samples:
        logger.info(
            "- Build_samples         :\t"
            + utils.convert_time(time_samples - time_stack)
        )
        logger.info(
            "- Random forest (total) :\t"
            + utils.convert_time(time_random_forest - time_samples)
        )
        logger.info(
            "- Post-processing       :\t"
            + utils.convert_time(end_time - time_random_forest)
        )
    logger.info("***")
    logger.info("Max workers used for parallel tasks " + str(args.n_workers))


def process_pekel(args, slurp_manager, margin):
    """
    Processes a Pekel raster and applies compute_pekel_mask
    using slurpContextManager.

    Creates a usable water mask and checks if there are enough
    water pixels to proceed with classification.

    Returns
    -------
    local_mask_pekel : np.ndarray
    mask_pekel : list
        SLURP key of the generated mask
    not_enough_water_samples : bool
    """

    # ==============================
    # LOAD PEKEL
    # ==============================

    pekel_array, pekel_profile = read_and_get_profile(args.extracted_pekel)
    args.pekel_nodata = pekel_profile.get("nodata")

    input_profile = deepcopy(pekel_profile)
    output_profile = eo_utils.double_uint8_profile(
        [deepcopy(pekel_profile)]
    )
    # ==============================
    # COMPUTE PEKEL MASK
    # ==============================
    local_mask_pekel, mask_pekel0 = mp_n_to_m_images(
        inputs=[pekel_array[0]],
        image_height=input_profile["height"],
        image_width=input_profile["width"],
        output_profiles = [output_profile[0], output_profile[0]],
        output_keys=["pekel_mask"],
        func=compute_pekel_mask,
        func_parameters={
            "thresh_pekel": args.thresh_pekel,
            "hand_strict": args.hand_strict,
            "strict_thresh": args.strict_thresh,
            "no_pekel_filter": args.no_pekel_filter,
            "pekel_nodata": args.pekel_nodata,
        },
        context_manager=slurp_manager,
        stable_margin=margin,
        binary=True,
    )

    # ==============================
    # CHECK NUMBER OF WATER PIXELS
    # ==============================

    not_enough_water_samples = False

    if np.count_nonzero(local_mask_pekel) < args.nb_samples_water:
        not_enough_water_samples = True
        logger.warning(
            "** WARNING ** Not enough water samples found in Pekel: "
            "switching to NDWI threshold mode."
        )

    return local_mask_pekel, mask_pekel0, not_enough_water_samples


def process_hand(args, slurp_manager, margin):
    """
    Processes a HAND (Height Above Nearest Drainage) raster
    using slurpContextManager and applies compute_hand_mask
    to create a usable mask.

    Parameters
    ----------
    args : Namespace
    slurp_manager : slurpContextManager
    margin : int

    Returns
    -------
    mask_hand : list
        SLURP key of the generated HAND mask
    """

    # ==============================
    # LOAD HAND
    # ==============================

    hand_array, hand_profile = read_and_get_profile(args.extracted_hand)

    input_profile = deepcopy(hand_profile)
    output_profile = eo_utils.single_float_profile(
        [deepcopy(hand_profile)]
    )

    # ==============================
    # COMPUTE HAND MASK
    # ==============================
    mask_hand = mp_n_to_m_images(
        inputs=[hand_array[0]],
        image_height=input_profile["height"],
        image_width=input_profile["width"],
        output_profiles=[output_profile],
        output_keys=["hand_mask"],
        func=compute_hand_mask,
        func_parameters={
            "thresh_hand": args.thresh_hand
        },
        context_manager=slurp_manager,
        stable_margin=margin,
        binary=False,
    )

    return mask_hand


def nominal_case_predict(
    args,
    slurp_manager,
    ndvi,
    ndwi,
    phr,
    valid_stack,
    local_mask_pekel,
    margin,
    mask_hand,
    mask_pekel0,
    phr_profile
):
    """
    Performs supervised classification using Random Forest to predict a water mask.
    """
    keys_files_layers = [
        read(raster_path=args.files_layers[i])
        for i in range(len(args.files_layers))
    ]
    
    nb_valid_pixels = len(
        np.where(valid_stack[0][0] == 0)[0]
    )
    args.nb_valid_water_pixels = np.count_nonzero(
        np.logical_and(local_mask_pekel, valid_stack[0][0] == 0)
    )
    input_profile = deepcopy(phr_profile)
    output_profile = eo_utils.single_uint8_profile(
        [deepcopy(phr_profile)]
    )

    args.nb_valid_other_pixels = nb_valid_pixels - args.nb_valid_water_pixels
    input_for_samples = [
        valid_stack[0][0],
        mask_hand[0],
        local_mask_pekel,
        phr[0][0],
        phr[0][1],
        phr[0][2],
        phr[0][3],
        ndvi[0][0],
        ndwi[0][0],
    ] + keys_files_layers
    samples = mp_n_to_m_scalars(
        inputs=input_for_samples,
        image_height=input_profile["height"],
        image_width=input_profile["width"],
        context_manager=slurp_manager,
        func=build_samples,
        func_parameters={
            "ndwi_threshold": args.ndwi_threshold,
            "nb_valid_water_pixels": args.nb_valid_water_pixels,
            "nb_valid_other_pixels": args.nb_valid_other_pixels,
            "nb_samples_water": args.nb_samples_water,
            "nb_samples_other": args.nb_samples_other,
            "samples_method": args.samples_method,
            "nb_samples_auto": args.nb_samples_auto,
            "auto_pct": args.auto_pct,
            "smart_area_pct": args.smart_area_pct,
            "smart_minimum": args.smart_minimum,
        },
        reducer=utils.concatenate_samples,
    )
    # samples=[x_samples, y_samples]
    del local_mask_pekel
    time_samples = time.time()
    # --Train classifier from samples-- #
    classifier = RandomForestClassifier(
        n_estimators=args.nb_estimators,
        max_depth=args.max_depth,
        random_state=712,
        n_jobs=1,
    )
    logger.debug(
        "RandomForest parameters: \n%s\n", str(classifier.get_params())
    )
    samples = np.concatenate(samples[:])
    x_samples = samples[:, 1:]  # im_phr, im_ndvi, im_ndwi and files_layers
    y_samples = samples[:, 0]  # mask_pekel
    rf_utils.train_classifier(classifier, x_samples, y_samples)
    rf_utils.print_feature_importance(classifier, args.files_layers)
    gc.collect()
    utils.display_mem_usage(args.debug, "After training step")
    # --Predict-- #
    input_for_prediction = [
        valid_stack[0][0],
        phr[0][0],
        phr[0][1],
        phr[0][2],
        phr[0][3],
        ndvi[0][0],
        ndwi[0][0],
    ] + keys_files_layers
    predict = mp_n_to_m_images(
        inputs=input_for_prediction,
        image_height=input_profile["height"],
        image_width=input_profile["width"],
        output_profiles=[output_profile],
        output_keys=["rf_prediction"],
        func=rf_prediction,
        func_parameters={
            "classifier": classifier,
            "debug": args.debug,
        },
        context_manager=slurp_manager,
        stable_margin=margin,
    )
    time_random_forest = time.time()
    utils.display_mem_usage(args.debug, "After prediction step")
    return predict, output_profile, time_random_forest, time_samples


def launch_postprocess(
    args,
    slurp_manager,
    predict,
    valid_stack,
    margin,
    mask_hand,
    mask_pekel0,
    predict_profile
):
    """
    Combines the predicted mask with additional masks and the validity mask.
    Applies the custom `post_process` filter using Slurp executor.
    Writes the post-processed mask to args.watermask,
    and optionally the raw output in debug mode.
    """

    inputs_for_classif = [
        predict[0],   # predicted RF mask
        mask_hand[0],     # hand mask
        mask_pekel0,    # Pekel mask
        valid_stack[0][0],  # validity mask
    ]

    input_profile = deepcopy(predict_profile)
    output_profile = eo_utils.double_uint8_profile([deepcopy(predict_profile)])

    # ---- Slurp executor version of n_images_to_m_images_filter ----
    im_predict, im_classif = mp_n_to_m_images(
        inputs=inputs_for_classif,
        image_height=input_profile["height"],
        image_width=input_profile["width"],
        output_profiles=[output_profile[0], output_profile[0]],
        output_keys=["postprocessed_mask", "raw_prediction"],
        func=post_process,
        func_parameters={
            "hand_filter": args.hand_filter,
            "hand_strict": args.hand_strict,
            "no_pekel_filter": args.no_pekel_filter,
            "binary_closing": args.binary_closing,
            "binary_opening": args.binary_opening,
            "area_closing": args.area_closing,
            "remove_small_holes": args.remove_small_holes,
            "remove_small_objects": args.remove_small_objects,
            "value_classif": args.value_classif,
        },
        context_manager=slurp_manager,
        stable_margin=margin,
    )
    # ---- write final mask ----
    slurp_manager.write_tif(
        data=im_classif, 
        path=args.watermask, 
        target_profile=output_profile[0]
    )

    # ---- write raw prediction in debug mode ----
   # if args.save_mode == "debug":
    slurp_manager.write_tif(
        data=im_predict,
        path=args.watermask.replace(".tif", "_raw_predict.tif"),
        target_profile=output_profile[0]
    )


def getarguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Compute Water Mask.")

    parser.add_argument(
        "main_config", help="First JSON file, load basis arguments"
    )

    parser.add_argument(
        "-log_f",
        "--logs_to_file",
        action="store_true",
        help="Store all logs to a file, instead of stdout",
    )
    parser.add_argument(
        "-d",
        "--debug",
        default=None,
        action="store_true",
        dest="debug",
        help="Debug flag",
    )

    group1 = parser.add_argument_group(description="*** INPUT FILES ***")
    group1.add_argument(
        "-user_config",
        help="Second JSON file, overload basis arguments if keys are the same",
    )
    group1.add_argument("-file_vhr", help="Input 4 bands VHR image")
    group1.add_argument("-valid", dest="valid_stack", help="Validity mask")
    group1.add_argument("-ndvi", dest="file_ndvi", help="NDVI filename")
    group1.add_argument("-ndwi", dest="file_ndwi", help="NDWI filename")
    group1.add_argument(
        "-pekel", dest="extracted_pekel", help="Extracted Pekel filename"
    )
    group1.add_argument(
        "-hand", dest="extracted_hand", help="Extracted Hand filename"
    )
    group1.add_argument(
        "-layers",
        nargs="+",
        dest="files_layers",
        help="Add layers as additional features used by learning algorithm",
    )
    group1.add_argument(
        "-filters",
        nargs="+",
        dest="file_filters",
        help="Add files used in filtering (postprocessing)",
    )

    group2 = parser.add_argument_group(description="*** OPTIONS ***")
    group2.add_argument(
        "-thresh_pekel", type=float, help="Pekel Threshold float"
    )
    group2.add_argument(
        "-hand_strict",
        action="store_true",
        help="Use not(pekelxx) for other (no water) samples",
    )
    group2.add_argument(
        "-thresh_hand", type=int, help="Hand Threshold int >= 0"
    )
    group2.add_argument(
        "-strict_thresh",
        type=float,
        help="Pekel Threshold float if hand_strict",
    )
    group2.add_argument(
        "-save_mode",
        choices=["none", "debug"],
        help="Save all files (debug) or only output mask (none)",
    )
    group2.add_argument(
        "-simple_ndwi_threshold",
        help="Compute water mask as a simple NDWI threshold - "
        "useful in arid places where no water is known by Peckel",
    )
    group2.add_argument(
        "-ndwi_threshold",
        type=float,
        help="Threshold used when Pekel is empty in the area",
    )

    group3 = parser.add_argument_group(
        description="*** LEARNING SAMPLES SELECTION AND CLASSIFIER ***"
    )
    group3.add_argument(
        "-samples_method",
        choices=["smart", "grid", "random"],
        help="Select method for choosing learning samples",
    )
    group3.add_argument(
        "-nb_samples_water",
        type=int,
        help="Number of samples in water for learning",
    )
    group3.add_argument(
        "-nb_samples_other",
        type=int,
        help="Number of samples in other for learning",
    )
    group3.add_argument(
        "-nb_samples_auto",
        action="store_true",
        help="Auto select number of samples for water and other",
    )
    group3.add_argument(
        "-auto_pct",
        type=float,
        help="Percentage of samples points, to use with -nb_samples_auto",
    )
    group3.add_argument(
        "-smart_area_pct",
        type=int,
        help=(
            "For smart method, importance of area when selecting the number of samples "
            "in each water surface"
        ),
    )
    group3.add_argument(
        "-smart_minimum",
        type=int,
        help="For smart method, minimum number of samples in each water surface.",
    )
    group3.add_argument(
        "-grid_spacing",
        type=int,
        help=(
            "For grid method, select samples on a regular grid "
            "(40 pixels seems to be a good value)"
        ),
    )
    group3.add_argument("-max_depth", type=int, help="Max depth of trees")
    group3.add_argument(
        "-nb_estimators", type=int, help="Nb of trees in Random Forest"
    )
    group3.add_argument(
        "-n_jobs",
        type=int,
        help="Nb of parallel jobs for Random Forest "
        "(1 is recommanded : use n_workers to optimize parallel computing)",
    )

    group4 = parser.add_argument_group(description="*** POST PROCESSING ***")
    group4.add_argument(
        "-no_pekel_filter",
        action="store_true",
        help="Deactivate postprocess with pekel which only keeps surfaces already known by pekel",
    )
    group4.add_argument(
        "-hand_filter",
        action="store_true",
        help="Postprocess with Hand (set to 0 when hand > thresh), incompatible with hand_strict",
    )
    group4.add_argument(
        "-binary_closing", type=int, help="Size of disk structuring element"
    )
    group4.add_argument(
        "-binary_opening", type=int, help="Size of disk structuring element"
    )
    group4.add_argument(
        "-area_closing",
        type=int,
        help="Area closing removes all dark structures",
    )
    group4.add_argument(
        "-remove_small_holes",
        type=int,
        help="The maximum area, in pixels, of a contiguous hole that will be filled",
    )

    group4.add_argument(
        "-remove_small_objects",
        type=int,
        help="The minimum area, in pixels, of a water body to be kept",
    )

    group5 = parser.add_argument_group(description="*** OUTPUT FILE ***")
    group5.add_argument("-watermask", help="Output classification filename")
    group5.add_argument(
        "-value_classif",
        type=int,
        help="Output classification value (default is 1)",
    )

    group6 = parser.add_argument_group(description="*** PARALLEL COMPUTING ***")
    group6.add_argument(
        "-n_workers", type=int, action="store", help="Number of CPU"
    )
    group6.add_argument(
        "-tile_max_size",
        type=int,
        help="Max tile size to be processed (0 : default)",
    )
    group6.add_argument(
        "-multiproc_context",
        default="spawn",
        help="Multiprocessing strategy: 'fork' or 'spawn' for EOScale",
    )
    args = parser.parse_args()

    utils.store_arglist(parser)

    return vars(args)


# --Main function-- #


def slurp_watermask(
    main_config: str,
    logs_to_file: bool,
    debug: bool,
    user_config: str,
    file_vhr: str,
    valid_stack: bool,
    file_ndvi: str,
    file_ndwi: str,
    extracted_pekel: str,
    extracted_hand: str,
    files_layers: list,
    file_filters: list,
    thresh_pekel: float,
    hand_strict: bool,
    thresh_hand: int,
    strict_thresh: float,
    save_mode: str,
    simple_ndwi_threshold: bool,
    ndwi_threshold: float,
    samples_method: str,
    nb_samples_water: int,
    nb_samples_other: int,
    nb_samples_auto: bool,
    auto_pct: float,
    smart_area_pct: int,
    smart_minimum: int,
    grid_spacing: int,
    max_depth: int,
    nb_estimators: int,
    n_jobs: int,
    no_pekel_filter: bool,
    hand_filter: bool,
    binary_closing: int,
    binary_opening: int,
    area_closing: int,
    remove_small_holes: int,
    remove_small_objects: int,
    watermask: str,
    value_classif: int,
    n_workers: int,
    tile_max_size: int,
    multiproc_context: str,
):
    """
    Main API to compute water mask using slurpContextManager
    (full shared memory mode).
    """

    keys = [
        "input",
        "aux_layers",
        "masks",
        "resources",
        "post_process",
        "water",
    ]

    argsdict, cli_params = utils.parse_args(keys, logs_to_file, main_config)

    for param in cli_params:
        if locals()[param] is not None:
            argsdict[param] = locals()[param]

    logger.info("--" * 50)
    logger.info("SLURP - Water mask\n")
    logger.info(f"JSON data loaded: {main_config}")

    args = argparse.Namespace(**argsdict)

    if args.debug:
        logger.handlers[0].setLevel(logging.DEBUG)

    logger.debug(f"{argsdict=}")

    # ==============================
    # SLURP CONTEXT
    # ==============================

    params = {
        "nb_max_workers": args.n_workers,
        "developer_mode": args.debug,
        "method": "mem",
        "mp_context": multiproc_context,
        "output_dir": path.dirname(args.file_vhr),
    }

    with slurpContextManager(params, tile_mode=True) as slurp_manager:

        try:
            t0 = time.time()

            # ==============================
            # LOAD INPUTS
            # ==============================
            logger.info("Step 0: Loading VHR image")

            # ==============================
            # BUILD STACK
            # ==============================

            ndvi, ndwi, phr, valid_stack, margin, phr_profile = (
                build_stack_water(args, slurp_manager)
            )

            # ==============================
            # PEKEL & HAND
            # ==============================
            logger.info("[1] Step: PEKEL")
            local_mask_pekel, mask_pekel0, not_enough_water_samples = (
                process_pekel(args, slurp_manager, margin)
            )
            logger.info("[2] Step: HAND")
            mask_hand = process_hand(args, slurp_manager, margin)

            # ==============================
            # SIMPLE NDWI THRESHOLD
            # ==============================

            if args.simple_ndwi_threshold:
                # TODO : NDWI PROFILE AND ARRAY
                # modif build stack water function to return profile and arrays separately 
                predict = mp_n_to_m_images(
                    inputs=[ndwi[0], valid_stack[0]],
                    image_height=input_profile["height"],
                    image_width=input_profile["width"],
                    output_profiles=[eo_utils.single_uint8_profile([input_profile])],
                    output_keys=[path.basename(args.watermask)],
                    func=utils.compute_mask_threshold,
                    func_parameters={"threshold": args.ndwi_threshold},
                    context_manager=slurp_manager,
                    stable_margin=margin,
                    binary=True,
                )

            # ==============================
            # RANDOM FOREST MODE
            # ==============================

            elif not_enough_water_samples:
                # TODO : NDWI PROFILE AND ARRAY
                # modif build stack water function to return profile and arrays separately 
                predict = mp_n_to_m_images(
                    inputs=[key_ndwi[0], key_valid_stack[0]],
                    image_height=input_profile["height"],
                    image_width=input_profile["width"],
                    output_profiles=[eo_utils.single_uint8_profile([input_profile])],
                    output_keys=[path.basename(args.watermask)],
                    func=utils.compute_mask_threshold,
                    func_parameters={"threshold": 1000},
                    context_manager=slurp_manager,
                    stable_margin=margin,
                    binary=True,
                )

            else:

                predict, predict_profile, time_random_forest, time_samples = nominal_case_predict(
                    args,
                    slurp_manager,
                    ndvi,
                    ndwi,
                    phr,
                    valid_stack,
                    local_mask_pekel,
                    margin,
                    mask_hand,
                    mask_pekel0,
                    phr_profile
                )

            # ==============================
            # POST PROCESS
            # ==============================

            launch_postprocess(
                args,
                slurp_manager,
                predict,
                valid_stack,
                margin,
                mask_hand,
                mask_pekel0,
                predict_profile
            )

            t1 = time.time()

            logger.info(
                "Total time (user):\t" + utils.convert_time(t1 - t0)
            )

        except Exception:
            logger.error("Unexpected error:", exc_info=True)
            traceback.print_exc()

    logger.info("End of watermask step\n")

def main():
    """
    Main function to run the water mask computation.
    It parses the command line arguments and calls the slurp_watermask function.
    """
    args = getarguments()
    slurp_watermask(**args)


if __name__ == "__main__":
    main()
