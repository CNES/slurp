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
import time
import traceback
import logging
import json
from os import makedirs, path

import eoscale.eo_executors as eoexe
import eoscale.manager as eom
import numpy as np
from skimage.measure import label, regionprops
from sklearn.ensemble import RandomForestClassifier

from slurp.post_process.morphology import apply_morpho
from slurp.tools import random_forest_utils as rf_utils
from slurp.tools import eoscale_utils as eo_utils
from slurp.tools import utils
from slurp.tools.constant import NODATA_INT8

logger = logging.getLogger("slurp")

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ModuleNotFoundError:
    logger.error("Intel(R) Extension/Optimization for scikit-learn not found.")


def compute_pekel_mask(
        input_buffer: list, input_profiles: list, params: dict
) -> list:
    """
    Compute Pekel mask regarding entry arguments

    :param list input_buffer: Pekel image [pekel_image]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: Pekel masks
    """
    mask_pekel = utils.compute_mask(input_buffer[0], [params["thresh_pekel"]])
    if params["hand_strict"]:
        mask_pekelxx = utils.compute_mask(
            input_buffer[0], [params["strict_thresh"]]
        )
        return [mask_pekel, mask_pekelxx]

    if not params["no_pekel_filter"]:
        mask_pekel0 = utils.compute_mask(input_buffer[0], [0])
    else:
        mask_pekel0 = np.zeros(input_buffer[0].shape)

    return [mask_pekel, mask_pekel0]


def compute_hand_mask(
        input_buffer: list, input_profiles: list, params: dict
) -> bool:
    """
    Compute Hand mask with one or multiple threshold values.

    :param list input_buffer: Hand image [hand_image]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: Hand mask (true if pixels are below a "thresh_hand" altitude)
    """
    mask_hand = input_buffer[0] > params["thresh_hand"]

    # Do not learn in water surface (useful if image contains big water surfaces)
    # Add some robustness if hand_strict is not used
    # if args.hand_strict:
    # np.logical_not(np.logical_or(mask_hand, inputBuffer[1]), out=mask_hand)
    # else:
    # np.logical_not(mask_hand, out=mask_hand)
    np.logical_not(mask_hand, out=mask_hand)

    return mask_hand


def get_random_indexes_from_masks(nb_indexes, mask_1, mask_2):
    """
    Get random valid indexes from masks.
    Mask 1 is a validity mask
    """
    np.random.seed(712)  # reproductible results
    rows_idxs = []
    cols_idxs = []

    if nb_indexes != 0:
        nb_idxs = 0

        height = mask_1.shape[0]
        width = mask_1.shape[1]

        while nb_idxs < nb_indexes:
            row = np.random.randint(0, height)
            col = np.random.randint(0, width)

            if mask_1[row, col] and mask_2[row, col]:
                rows_idxs.append(row)
                cols_idxs.append(col)
                nb_idxs += 1

    return rows_idxs, cols_idxs


def get_grid_indexes_from_mask(nb_samples, valid_mask, mask_ground_truth):
    """
    Retrieve row and columns indices selected on the valid pixel of the image

    :param int nb_samples: number of samples selected
    :param boolean numpy array valid_mask :
    :param boolean numpy array mask_ground_truth :
    :return: tuple of list , row indices and columns indices
    """
    valid_samples = np.logical_and(mask_ground_truth, valid_mask).astype(
        np.uint8
    )
    _, rows, cols = np.where(valid_samples)

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


def get_smart_indexes_from_mask(nb_indexes, pct_area, minimum, mask):
    """
    Retrieve row and columns indices selected on the valid pixel of the image

    :param int nb_indexes: number of samples selected
    :param int pct_area: importance of area for selecting number of samples in each water surface
    :param int minimum: minimum number of samples in each water surface
    :param boolean numpy array mask:
    :return: tuple of list , row indices and columns indices

    """
    rows_idxs = []
    cols_idxs = []

    if nb_indexes != 0:
        img_labels, nb_labels = label(mask, return_num=True)
        props = regionprops(img_labels)
        mask_area = float(np.sum(mask))

        # number of samples for each label/prop
        n1_indexes = int((1.0 - pct_area / 100.0) * nb_indexes / nb_labels)

        # number of samples to distribute to each label/prop
        n2_indexes = pct_area / 100.0 * nb_indexes / mask_area

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
        input_buffer: list, input_profiles: list, params: dict
) -> np.ndarray:
    """
    Build samples

    :param list input_buffer: [valid_stack, mask_hand, mask_pekel, im_phr, im_ndvi, im_ndwi] + files_layers
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: Retrieve number of pixels for each class
    """
    nb_valid_subset = np.count_nonzero(input_buffer[0])
    valid_water_pixels = np.logical_and(
        input_buffer[2], input_buffer[5] > params["ndwi_threshold"]
    )

    nb_water_subset = np.count_nonzero(
        np.logical_and(valid_water_pixels, input_buffer[0])
    )

    # valid pixels for 'other' (every thing but water) : valid mask + hand > 0
    # This criteria should be reconsidered (ie : think of a relative threshold to select samples not too far from water)
    nb_valid_pix_other = np.count_nonzero(
        np.logical_and(input_buffer[0], input_buffer[1])
    )
    nb_other_subset = nb_valid_pix_other

    # Ratio of pixel class compare to the full image ratio
    water_ratio = nb_water_subset / params["nb_valid_water_pixels"]
    other_ratio = nb_other_subset / params["nb_valid_other_pixels"]
    # Retrieve number of samples to create for each class in this subset
    nb_water_subsamples = round(water_ratio * params["nb_samples_water"])
    nb_other_subsamples = round(other_ratio * params["nb_samples_other"])

    # Prepare random water and other samples
    if params["nb_samples_auto"]:
        nb_water_subsamples = int(nb_water_subset * params["auto_pct"])
        nb_other_subsamples = int(nb_other_subset * params["auto_pct"])

    # Pekel samples
    if params["samples_method"] == "random":
        rows_pekel, cols_pekel = get_random_indexes_from_masks(
            nb_water_subsamples, input_buffer[0][0], input_buffer[2][0]
        )
        # Hand samples, always random (currently)
        rows_hand, cols_hand = get_random_indexes_from_masks(
            nb_other_subsamples, input_buffer[0][0], input_buffer[1][0]
        )

    elif params["samples_method"] == "smart":
        rows_pekel, cols_pekel = get_smart_indexes_from_mask(
            nb_water_subsamples,
            params["smart_area_pct"],
            params["smart_minimum"],
            np.logical_and(input_buffer[2][0], input_buffer[0][0]),
        )
        # Hand samples, always random (currently)
        rows_hand, cols_hand = get_random_indexes_from_masks(
            nb_other_subsamples, input_buffer[0][0], input_buffer[1][0]
        )

    elif params["samples_method"] == "grid":
        rows_pekel, cols_pekel = get_grid_indexes_from_mask(
            nb_water_subsamples, input_buffer[0], valid_water_pixels[0]
        )

        # Hand samples, always random (currently)
        rows_hand, cols_hand = get_random_indexes_from_masks(
            nb_other_subsamples, input_buffer[0][0], input_buffer[1][0]
        )

    else:
        raise Exception(
            "Sample method not accepted : use 'random', 'smart' or 'grid'"
        )

    # All samples
    rows = np.concatenate((rows_pekel, rows_hand))
    cols = np.concatenate((cols_pekel, cols_hand))

    # Prepare samples for learning
    im_stack = np.concatenate((input_buffer[2:]), axis=0)
    samples = np.transpose(
        im_stack[:, rows.astype(np.uint16), cols.astype(np.uint16)]
    )

    return samples  # [x_samples, y_samples]


def rf_prediction(
        input_buffer: list, input_profiles: list, params: dict
) -> np.ndarray:
    """
    Random Forest prediction

    :param list input_buffer: [key_valid_stack, key_phr, key_ndvi, key_ndwi] + file_layers
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: predicted mask
    """
    im_stack = np.concatenate((input_buffer[1:]), axis=0)
    valid_mask = input_buffer[0].astype(bool)
    buffer_to_predict = np.transpose(im_stack[:, valid_mask[0].astype(bool)])

    classifier = params["classifier"]
    prediction = np.zeros(valid_mask[0].shape, dtype=np.uint8)
    if buffer_to_predict.shape[0] > 0:  # not only NO DATA
        prediction[valid_mask[0].astype(bool)] = classifier.predict(
            buffer_to_predict
        )

    utils.display_mem_usage(
        params["debug"],
        f"RF Prediction on buffer {input_buffer[0].shape[1]} x {input_buffer[0].shape[2]}",
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
    logger.info(
        "Simple threshold mask NDWI > " + str(args.ndwi_threshold)
    )
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
        input_buffer: list, input_profiles: list, params: dict
) -> list:
    """
    Compute some filters on the prediction image.

    :param list input_buffer: [im_predict, mask_hand, mask_pekel0, valid_stack]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: predict mask and post-processed mask
    """
    buffer_shape = input_buffer[0].shape

    # Filter with Hand
    if params["hand_filter"]:
        if not params["hand_strict"]:
            input_buffer[0][np.logical_not(input_buffer[1])] = 0
        else:
            logger.warning("\nWARNING: hand_filter and hand_strict are incompatible.")

    # Filter for final classification
    if not params["no_pekel_filter"]:
        mask = np.zeros(buffer_shape, dtype=bool)
        if not params["no_pekel_filter"]:  # filter with pekel0
            mask = np.zeros(buffer_shape, dtype=bool)
            mask = np.logical_or(
                mask, input_buffer[2][0]
            )  # problème de mask_pekel0 if "not defined"
        im_classif = mask_filter(input_buffer[0], mask)
    else:
        im_classif = input_buffer[0]

    # Closing
    if params["binary_closing"]:
        im_classif[0, :, :] = apply_morpho(
            im_classif[0, :, :].astype(bool),
            "binary_closing",
            params["binary_closing"],
        ).astype(np.uint8)
    if params["area_closing"]:
        im_classif[0, :, :] = apply_morpho(
            im_classif[0, :, :], "area_closing", params["area_closing"]
        )
    if params["remove_small_holes"]:
        im_classif[0, :, :] = apply_morpho(
            im_classif[0, :, :].astype(bool),
            "remove_small_holes",
            params["remove_small_holes"],
        ).astype(np.uint8)

    # Add nodata in im_classif
    im_classif[np.logical_not(input_buffer[3])] = NODATA_INT8
    im_classif[im_classif == 1] = params["value_classif"]

    im_predict = input_buffer[0]
    im_predict[np.logical_not(input_buffer[3])] = NODATA_INT8
    im_predict[im_predict == 1] = params["value_classif"]

    return [im_predict, im_classif]


def build_stack_water(args, eoscale_manager):
    """
    Prepares and returns the required image layers and masks for further processing.

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
    eoscale_manager : object
        Image manager handling raster reading and metadata extraction.
    """
    # Image PHR (numpy array, 4 bands, band number is first dimension),
    key_phr = eoscale_manager.open_raster(raster_path=args.file_vhr)
    profile_phr = eoscale_manager.get_profile(key_phr)
    args.nodata_phr = profile_phr["nodata"]
    args.shape = (profile_phr["height"], profile_phr["width"])
    args.crs = profile_phr["crs"]
    args.transform = profile_phr["transform"]
    args.rpc = None
    # Valid stack
    key_valid_stack = eoscale_manager.open_raster(
        raster_path=args.valid_stack
    )
    # margin for masks computation (a Pekel pixel is 30m large, ~ 45m in its diagonal
    # for 0.5 m images, a 100 pixels margin will allow to cover one Pekel pixel
    margin = 100
    # NDXI
    key_ndvi = eoscale_manager.open_raster(raster_path=args.file_ndvi)
    key_ndwi = eoscale_manager.open_raster(raster_path=args.file_ndwi)
    return key_ndvi, key_ndwi, key_phr, key_valid_stack, margin


def display_global_infos(args, end_time, t0, time_stack):
    """
    Displays general information about the water mask processing.
    """
    logger.info(
        f"**** Water mask for {args.file_vhr} (saved as {args.watermask}) ****"
    )
    logger.info(
        "Total time (user)       :\t"
        + utils.convert_time(end_time - t0)
    )
    logger.info(
        "- Build_stack           :\t"
        + utils.convert_time(time_stack - t0)
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


def display_computation_info(args, end_time, not_enough_water_samples, t0, time_random_forest, time_samples,
                             time_stack):
    """
    Displays information about the entire computation process, including when handling edge cases.
    """
    logger.info(
        f"**** Water mask for {args.file_vhr} (saved as {args.watermask}) ****"
    )
    logger.info(
        "Total time (user)       :\t"
        + utils.convert_time(end_time - t0)
    )
    logger.info(
        "- Build_stack           :\t"
        + utils.convert_time(time_stack - t0)
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


def process_pekel(args, eoscale_manager, margin):
    """
    Processes a Pekel (water) raster and applies the `compute_pekel_mask` filter to create a usable water mask.
    Checks if there are enough water pixels in the Pekel mask to proceed with classification.
    If not enough water pixels are found, the function flags this and suggests using NDWI thresholding instead.
    """
    key_pekel = eoscale_manager.open_raster(
        raster_path=args.extracted_pekel
    )
    pekel_profile = eoscale_manager.get_profile(key_pekel)
    args.pekel_nodata = pekel_profile["nodata"]
    # Pekel valid masks
    mask_pekel = eoexe.n_images_to_m_images_filter(
        inputs=[key_pekel],
        image_filter=compute_pekel_mask,
        filter_parameters=vars(args),
        generate_output_profiles=eo_utils.double_uint8_profile,
        stable_margin=margin,
        context_manager=eoscale_manager,
        multiproc_context=args.multiproc_context,
        filter_desc="Pekel valid mask processing...",
    )
    # If user wants a simple threshold on NDWI values, we don't select samples and launch learning/prediction step
    # If there are not enough water samples, we return a void mask
    not_enough_water_samples = False
    # Check pekel mask
    # - if there are too few values : we threshold NDWI to detect water areas
    # - if there are even no "supposed water areas" : stop machine learning process (flag select_samples=False)
    local_mask_pekel = eoscale_manager.get_array(mask_pekel[0])
    if np.count_nonzero(local_mask_pekel) < args.nb_samples_water:
        # In case they are too few Pekel pixels, we prefer to threshold NDWI and skip samples selection
        # Alternative would be to select samples in a thresholded NDWI...
        not_enough_water_samples = True
        logger.warning(
            "** WARNING ** not enough water samples are found in Pekel : return a void mask"
        )
    return local_mask_pekel, mask_pekel, not_enough_water_samples


def process_hand(args, eoscale_manager, margin):
    """
    Processes a HAND (Height Above Nearest Drainage) raster and applies the `compute_hand_mask` filter to create a usable mask for further processing.
    """
    key_hand = eoscale_manager.open_raster(
        raster_path=args.extracted_hand
    )
    hand_profile = eoscale_manager.get_profile(key_hand)
    args.hand_nodata = hand_profile["nodata"]
    # Create HAND mask
    mask_hand = eoexe.n_images_to_m_images_filter(
        inputs=[key_hand],
        image_filter=compute_hand_mask,
        # args.hand_strict impossible because of mask_pekel0 not sure
        filter_parameters=vars(args),
        generate_output_profiles=eo_utils.single_float_profile,
        stable_margin=margin,
        context_manager=eoscale_manager,
        multiproc_context=args.multiproc_context,
        filter_desc="Hand valid mask processing...",
    )
    return mask_hand


def nominal_case_predict(args, eoscale_manager, key_ndvi, key_ndwi, key_phr, key_valid_stack, local_mask_pekel, margin,
                         mask_hand, mask_pekel):
    """
    Performs supervised classification using Random Forest to predict a water mask.
    """
    keys_files_layers = [
        eoscale_manager.open_raster(
            raster_path=args.files_layers[i]
        )
        for i in range(len(args.files_layers))
    ]
    # Sample selection
    valid_stack = eoscale_manager.get_array(key_valid_stack)
    nb_valid_pixels = np.count_nonzero(valid_stack)
    args.nb_valid_water_pixels = np.count_nonzero(
        np.logical_and(local_mask_pekel, valid_stack)
    )
    args.nb_valid_other_pixels = (
            nb_valid_pixels - args.nb_valid_water_pixels
    )
    input_for_samples = [
                            key_valid_stack,
                            mask_hand[0],
                            mask_pekel[0],
                            key_phr,
                            key_ndvi,
                            key_ndwi,
                        ] + keys_files_layers
    samples = eoexe.n_images_to_m_scalars(
        inputs=input_for_samples,
        image_filter=build_samples,
        filter_parameters=vars(args),
        nb_output_scalars=args.nb_samples_water
                          + args.nb_samples_other,
        context_manager=eoscale_manager,
        concatenate_filter=eo_utils.concatenate_samples,
        output_scalars=[],
        multiproc_context=args.multiproc_context,
        filter_desc="Samples water processing...",
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
    samples = np.concatenate(samples[:])  # A revoir si possible
    x_samples = samples[
                :, 1:
                ]  # im_phr, im_ndvi, im_ndwi and files_layers
    y_samples = samples[:, 0]  # mask_pekel
    rf_utils.train_classifier(classifier, x_samples, y_samples)
    rf_utils.print_feature_importance(classifier, args.files_layers)
    gc.collect()
    utils.display_mem_usage(args.debug, "After training step")
    # --Predict-- #
    input_for_prediction = [
                               key_valid_stack,
                               key_phr,
                               key_ndvi,
                               key_ndwi,
                           ] + keys_files_layers
    key_predict = eoexe.n_images_to_m_images_filter(
        inputs=input_for_prediction,
        image_filter=rf_prediction,
        filter_parameters={
            "classifier": classifier,
            "debug": args.debug,
        },
        generate_output_profiles=eo_utils.single_uint8_profile,
        stable_margin=margin,
        context_manager=eoscale_manager,
        multiproc_context=args.multiproc_context,
        filter_desc="RF prediction processing...",
    )
    time_random_forest = time.time()
    utils.display_mem_usage(args.debug, "After prediction step")
    return key_predict, time_random_forest, time_samples


def launch_postprocess(args, eoscale_manager, key_predict, key_valid_stack, margin, mask_hand, mask_pekel):
    """
    Combines the predicted mask with additional masks and the validity mask. Then it applies a custom `post_process` filter.
    Finally, writes the post-processed mask to `args.watermask`, and optionally the raw output (when debug mode is activated).
    """
    inputs_for_classif = [
        key_predict[0],
        mask_hand[0],
        mask_pekel[1],
        key_valid_stack,
    ]
    im_classif = eoexe.n_images_to_m_images_filter(
        inputs=inputs_for_classif,
        image_filter=post_process,
        filter_parameters=vars(args),
        generate_output_profiles=eo_utils.double_uint8_profile,
        stable_margin=margin,
        context_manager=eoscale_manager,
        multiproc_context=args.multiproc_context,
        filter_desc="Post processing...",
    )
    eoscale_manager.write(
        key=im_classif[1], img_path=args.watermask
    )  # classif
    if args.save_mode == "debug":
        eoscale_manager.write(
            key=im_classif[0],
            img_path=args.watermask.replace(
                ".tif", "_raw_predict.tif"
            ),
        )

def getarguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Compute Water Mask.")

    parser.add_argument(
        "main_config", help="First JSON file, load basis arguments"
    )

    parser.add_argument("-log_f",
                        "--logs_to_file",
                        action="store_true",
                        help="Store all logs to a file, instead of stdout",
                        )
    parser.add_argument(
        "-d", "--debug", default=None, action="store_true", dest="debug", help="Debug flag"
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
        help="For smart method, importance of area for selecting number of samples in each water surface",
    )
    group3.add_argument(
        "-smart_minimum",
        type=int,
        help="For smart method, minimum number of samples in each water surface.",
    )
    group3.add_argument(
        "-grid_spacing",
        type=int,
        help="For grid method, select samples on a regular grid (40 pixels seems to be a good value)",
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
        "-area_closing",
        type=int,
        help="Area closing removes all dark structures",
    )
    group4.add_argument(
        "-remove_small_holes",
        type=int,
        help="The maximum area, in pixels, of a contiguous hole that will be filled",
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


def slurp_watermask(main_config: str, logs_to_file: bool, debug: bool, user_config: str, file_vhr: str,
                    valid_stack: bool,
                    file_ndvi: str, file_ndwi: str, extracted_pekel: str, extracted_hand: str, files_layers: list,
                    file_filters: list, thresh_pekel: float, hand_strict: bool, thresh_hand: int, strict_thresh: float,
                    save_mode: str, simple_ndwi_threshold: bool, ndwi_threshold: float,
                    samples_method: str, nb_samples_water: int, nb_samples_other: int, nb_samples_auto: bool,
                    auto_pct: float, smart_area_pct: int, smart_minimum: int, grid_spacing: int,
                    max_depth: int, nb_estimators: int, n_jobs: int,
                    no_pekel_filter: bool, hand_filter: bool, binary_closing: int,
                    area_closing: int, remove_small_holes: int, watermask: str,
                    value_classif: int, n_workers: int, tile_max_size: int, multiproc_context: str):
    """
    Main API to compute water mask.
    """
    # Read the JSON files
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
        # If the parameter from the CLI is not None, we update argsdict with the value from the CLI
        if locals()[param] is not None:
            argsdict[param] = locals()[param]

    logger.info("--" * 50 + "\nSLURP - Water mask\n" + f"JSON data loaded: {main_config}")
    args = argparse.Namespace(**argsdict)
    if args.debug:
        logger.handlers[0].setLevel(logging.DEBUG)
    logger.debug(f"{argsdict=}")

    # Mask calculation
    with eom.EOContextManager(
            nb_workers=args.n_workers,
            tile_mode=True,
            tile_max_size=args.tile_max_size,
    ) as eoscale_manager:
        try:
            t0 = time.time()

            utils.display_mem_usage(args.debug, "Start computation")

            # --Build stack with all layers-- #

            key_ndvi, key_ndwi, key_phr, key_valid_stack, margin = build_stack_water(args, eoscale_manager)

            time_stack = time.time()

            # --Build samples-- #

            # Pekel
            local_mask_pekel, mask_pekel, not_enough_water_samples = process_pekel(args, eoscale_manager, margin)

            # HAND
            mask_hand = process_hand(args, eoscale_manager, margin)

            # Flag to command post-process
            do_post_process = True

            if args.simple_ndwi_threshold:
                # Simple NDWI threshold, but taking account valid stack to take care of NO_DATA values
                do_post_process, key_predict, time_random_forest, time_samples = apply_ndwi_thresh(args,
                                                                                                   eoscale_manager,
                                                                                                   key_ndwi,
                                                                                                   key_valid_stack)

            elif not_enough_water_samples:
                # We compute a void mask (0 everywhere, except for NO DATA values)
                # Tips : we threshold NDWI > 1000 : no pixel should be detected.
                key_predict = eoexe.n_images_to_m_images_filter(
                    inputs=[key_ndwi, key_valid_stack],
                    image_filter=utils.compute_mask_threshold,
                    filter_parameters={"threshold": 1000},
                    context_manager=eoscale_manager,
                    generate_output_profiles=eo_utils.single_uint8_profile,
                    multiproc_context=args.multiproc_context,
                    filter_desc="Void mask",
                )

                do_post_process = False

            else:
                # Nominal case : select samples, train, predict
                #
                # Taking optional layers into account
                key_predict, time_random_forest, time_samples = nominal_case_predict(args, eoscale_manager, key_ndvi,
                                                                                     key_ndwi, key_phr, key_valid_stack,
                                                                                     local_mask_pekel, margin,
                                                                                     mask_hand, mask_pekel)
            if do_post_process:
                # --Post_processing-- #
                launch_postprocess(args, eoscale_manager, key_predict, key_valid_stack, margin, mask_hand, mask_pekel)

            else:
                eoscale_manager.write(
                    key=key_predict[0], img_path=args.watermask
                )  # classif

            utils.display_mem_usage(args.debug, "End of computation")
            end_time = time.time()

            display_global_infos(args, end_time, t0, time_stack)
            if not args.simple_ndwi_threshold and not not_enough_water_samples:
                display_rf_infos(end_time, time_random_forest, time_samples, time_stack)

            logger.info("***")
            logger.info("Max workers used for parallel tasks " + str(args.n_workers))

        except FileNotFoundError as fnfe_exception:
            logger.error("FileNotFoundError", fnfe_exception)

        except PermissionError as pe_exception:
            logger.error("PermissionError", pe_exception)

        except ArithmeticError as ae_exception:
            logger.error("ArithmeticError", ae_exception)

        except MemoryError as me_exception:
            logger.error("MemoryError", me_exception)

        except Exception as exception:  # pylint: disable=broad-except
            logger.error("oups...", exception)
            traceback.print_exc()


def main():
    """
    Main function to run the water mask computation.
    It parses the command line arguments and calls the slurp_watermask function.
    """
    args = getarguments()
    slurp_watermask(**args)


if __name__ == "__main__":
    main()