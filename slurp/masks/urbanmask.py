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

"""Compute building and road masks from VHR images thanks to OSM layers"""

import argparse
import gc
import time
import traceback
import logging
import pathlib
import json
from os import makedirs, path, remove

import eoscale.eo_executors as eoexe
import eoscale.manager as eom
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from slurp.post_process.morphology import apply_morpho
from slurp.tools import random_forest_utils
from slurp.tools import eoscale_utils as eo_utils
from slurp.tools import io_utils, utils
from slurp.tools.constant import NODATA_INT8

logger = logging.getLogger("slurp")

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ModuleNotFoundError:
    logger.error("Intel(R) Extension/Optimization for scikit-learn not found.")



def apply_vegetationmask(
    input_buffer: list, input_profiles: list, params: dict
) -> np.ndarray:
    """
    Calculation of the valid pixels of a given image outside vegetation mask

    :param list input_buffer: VHR input image [valid_stack, vegetationmask]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments, must contain the keys "vegmask_min_value" and "veg_binary_dilation"
    :returns: valid_phr (boolean numpy array, True = valid data, False = no data)
    """
    non_veg = np.where(
        input_buffer[1] < params["vegmask_min_value"], True, False
    )
    # dilate non vegetation areas, because sometimes the vegetation mask can cover urban areas
    non_veg_dilated = apply_morpho(
        non_veg[0], "binary_dilation", params["veg_binary_dilation"]
    )
    valid_stack = np.logical_and(input_buffer[0], [non_veg_dilated])

    return valid_stack


def apply_watermask(
    input_buffer: list, input_profiles: list, params: dict
) -> np.ndarray:
    """
    Calculation of the valid pixels of a given image outside water mask

    :param list input_buffer: VHR input image [valid_stack, watermask]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments (not used but necessary for eoscale)
    :returns: valid_phr (boolean numpy array, True = valid data, False = no data)
    """
    valid_stack = np.logical_and(
        input_buffer[0], np.where(input_buffer[1] == 0, True, False)
    )

    return valid_stack


def get_grid_indexes_from_mask(nb_samples, valid_mask, mask_ground_truth):
    """
    Recover of row and columns indices selected on the valid pixel of the image

    :param int nb_samples:
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
        indices = np.arange(0, len(rows) - 1, int(len(rows) / nb_samples))
        s_rows = rows[indices]
        s_cols = cols[indices]
    else:
        s_rows = []
        s_cols = []

    return s_rows, s_cols


def build_samples(
    input_buffer: list, input_profiles: list, params: dict
) -> np.ndarray:
    """
    Build samples

    :param list input_buffer: [valid_stack, gt, im_phr, im_ndvi, im_ndwi] + files_layers
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: Retrieve number of pixels for each class
    """
    # Beware that WSF ground truth contains 0 (non building), 255 (building) but sometimes 1 (invalid pixels ?)
    mask_building_before_erosion = np.where(
        input_buffer[1] == params["value_classif"], True, False
    )
    mask_building = [
        apply_morpho(
            mask_building_before_erosion[0],
            "binary_erosion",
            params["gt_binary_erosion"],
        )
    ]
    mask_non_building = np.where(input_buffer[1] == 0, True, False)

    # Retrieve number of pixels for each class
    nb_built_subset = np.count_nonzero(
        np.logical_and(mask_building, input_buffer[0])
    )
    nb_other_subset = np.count_nonzero(
        np.logical_and(mask_non_building, input_buffer[0])
    )
    # Ratio of pixel class compare to the full image ratio
    urban_ratio = nb_built_subset / params["nb_valid_built_pixels"]
    other_ratio = nb_other_subset / params["nb_valid_other_pixels"]
    # Retrieve number of samples to create for each class in this subset
    nb_urban_subsamples = round(urban_ratio * params["nb_samples_urban"])
    nb_other_subsamples = round(other_ratio * params["nb_samples_other"])

    rows_b, cols_b = [], []
    rows_nob, cols_nob = [], []
    if nb_urban_subsamples > 0:
        # Building samples
        rows_b, cols_b = get_grid_indexes_from_mask(
            nb_urban_subsamples, input_buffer[0][0], mask_building
        )

        if nb_other_subsamples > 0:
            rows_nob, cols_nob = get_grid_indexes_from_mask(
                nb_other_subsamples, input_buffer[0][0], mask_non_building
            )
    else:

        if nb_other_subsamples > 0:
            rows_nob, cols_nob = get_grid_indexes_from_mask(
                nb_other_subsamples, input_buffer[0][0], mask_non_building
            )

    rows = np.concatenate((rows_b, rows_nob))
    cols = np.concatenate((cols_b, cols_nob))

    # Prepare samples for learning
    im_stack = np.concatenate((input_buffer[1:]), axis=0)
    samples = np.transpose(
        im_stack[:, rows.astype(np.uint16), cols.astype(np.uint16)]
    )

    return samples


def rf_prediction(
    input_buffer: list, input_profiles: list, params: dict
) -> list:
    """
    Random Forest prediction

    :param list input_buffer: [original_valid_stack (without water and veg masks), valid_stack, vhr_image, ndvi, ndwi] + file_layers
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: predicted mask (proba)
    """
    im_stack = np.concatenate((input_buffer[2:]), axis=0)
    nodata_mask = (1 - input_buffer[0]).astype(bool)
    valid_mask = input_buffer[1].astype(bool)
    buffer_to_predict = np.transpose(im_stack[:, valid_mask[0]])
    # buffer_to_predict are non NODATA pixels, defined by all the primitives (R-G-B-NIR-NDVI-NDWI-[+ features]

    classifier = params["classifier"]
    if buffer_to_predict.shape[0] > 0:
        proba = classifier.predict_proba(buffer_to_predict)
        # Prediction, inspired by sklearn code to predict class
        res_classif = classifier.classes_.take(np.argmax(proba, axis=1), axis=0)
        res_classif[res_classif == 255] = 1

        prediction = np.zeros(valid_mask.shape)
        prediction[0][valid_mask[0]] = res_classif
        prediction[0][nodata_mask[0]] = NODATA_INT8

        proba_buildings = np.zeros(valid_mask.shape)
        proba_buildings[0][valid_mask[0]] = (
            100 * proba[:, 1]
        )  # Proba for class 1 (buildings)
        proba_buildings[0][nodata_mask[0]] = NODATA_INT8

    else:
        # corner case : only NO_DATA !
        prediction = np.full(valid_mask.shape, NODATA_INT8)
        proba_buildings = np.full(valid_mask.shape, NODATA_INT8)

    return [proba_buildings, prediction]

def nominal_case_urbanmask(args, eoscale_manager, gt_key, key_ndvi, key_ndwi, key_original_valid_stack, key_phr,
                           key_valid_stack, keys_files_layers, t0, time_stack):
    """
    Generate an urban mask by training a classifier with extracted samples.

    If there are sufficient building and non-building samples, it trains a classifier to generate the urban mask.
    If there is not enough samples, an empty mask is produced.

    The output mask is written to disk at the location specified in `args.urbanmask`.

    Parameters
    ----------
    args : Namespace
        Configuration and parameters including thresholds, file paths, and multiprocessing context.
        Expected to have at least these attributes:
            - nb_valid_built_pixels (int)
            - nb_valid_other_pixels (int)
            - multiproc_context (optional)
            - value_classif (int): classification label for buildings
            - file_vhr (str): path or identifier for very high resolution input file
            - urbanmask (str): path where the resulting urban mask will be saved
    eoscale_manager : object
        Manager object handling image processing context and file writing.
    gt_key : object
        Key or reference to ground truth data layer.
    key_ndvi : object
        Key or reference to NDVI (Normalized Difference Vegetation Index) data layer.
    key_ndwi : object
        Key or reference to NDWI (Normalized Difference Water Index) data layer.
    key_original_valid_stack : object
        Key or reference to the original valid data stack.
    key_phr : object
        Key or reference to panchromatic high-resolution image layer.
    key_valid_stack : object
        Key or reference to valid data stack used for sampling.
    keys_files_layers : list
        List of additional image keys or layers used for sampling.
    t0 : float
        Initial timestamp or start time reference.
    time_stack : object
        Data structure representing time-series image stack or related temporal data.
    """
    input_for_samples = [
                            key_valid_stack,
                            gt_key,
                            key_phr,
                            key_ndvi,
                            key_ndwi,
                        ] + keys_files_layers
    samples = eoexe.n_images_to_m_scalars(
        inputs=input_for_samples,
        image_filter=build_samples,
        filter_parameters=vars(args),
        nb_output_scalars=args.nb_valid_built_pixels
                          + args.nb_valid_other_pixels,
        context_manager=eoscale_manager,
        concatenate_filter=eo_utils.concatenate_samples,
        output_scalars=[],
        multiproc_context=args.multiproc_context,
        filter_desc="Samples building processing...",
    )
    samples = np.concatenate(samples[:])
    x_samples = samples[
                :, 1:
                ]  # im_phr, im_ndvi, im_ndwi and files_layers
    y_samples = samples[:, 0]  # gt
    # Check if we have found "building" AND "non building" samples
    # (in very rare cases, WSF has only a small spot that is eroded in the build_samples step)
    building_areas = len(np.where(y_samples == args.value_classif)[0]) > 0
    non_building_areas = len(np.where(y_samples == 0)[0]) > 0
    time_samples = time.time()
    if building_areas and non_building_areas:
        # Train classifier from samples and predict urban mask
        time_random_forest = samples_train_and_predict(args, eoscale_manager, key_ndvi, key_ndwi, key_original_valid_stack,
                                 key_phr, key_valid_stack, keys_files_layers, x_samples, y_samples)
        end_time = time.time()
        display_logs_rf(args, end_time, t0, time_samples, time_stack, time_random_forest)
    else:
        # Weird corner case : learning/prediction had not enough samples
        logger.info(
            f"**** Corner case with too few urban samples for {args.file_vhr} -> void mask saved as {args.urbanmask} ****"
        )

        key_predict = eoexe.n_images_to_m_images_filter(
            inputs=[key_original_valid_stack],
            image_filter=add_nodata,
            filter_parameters={"fill_value": 0},
            generate_output_profiles=eo_utils.single_uint8_profile,
            context_manager=eoscale_manager,
            multiproc_context=args.multiproc_context,
            filter_desc="Add nodata...",
        )

        # Save proba mask
        eoscale_manager.write(
            key=key_predict[0], img_path=args.urbanmask
        )


def display_logs_rf(args, end_time, t0, time_samples, time_stack, time_random_forest):
    """
    Logs timing information and output paths for the urban mask generation pipeline.
    """
    logger.info(
        f"**** Urban proba mask for {args.file_vhr} (saved as {args.urbanmask}) ****"
    )
    logger.info(
        "Total time (user)       :\t"
        + utils.convert_time(end_time - t0)
    )
    logger.info(
        "- Build_stack           :\t"
        + utils.convert_time(time_stack - t0)
    )
    logger.info(
        "- Build_samples         :\t"
        + utils.convert_time(time_samples - time_stack)
    )
    logger.info(
        "- Random forest (total) :\t"
        + utils.convert_time(time_random_forest - time_samples)
    )
    logger.info("***")


def build_stack_urban(args, eoscale_manager):
    """
    Prepares and returns the required image layers and masks for further processing.

    Parameters
    ----------
    args : Namespace
        Argument namespace containing file paths and processing parameters.
        Expected attributes:
            - file_vhr : str
            - valid_stack : str
            - vegetationmask : str (optional)
            - watermask : str (optional)
            - file_ndvi : str
            - file_ndwi : str
            - extracted_wsf : str
            - multiproc_context : optional
    eoscale_manager : object
        Manager handling raster I/O and context for image operations.
    """
    # Image PHR (numpy array, 4 bands, band number is first dimension),
    key_phr = eoscale_manager.open_raster(raster_path=args.file_vhr)
    profile_phr = eoscale_manager.get_profile(key_phr)
    # Valid stack
    key_original_valid_stack = eoscale_manager.open_raster(
        raster_path=args.valid_stack
    )
    key_valid_stack = key_original_valid_stack
    # Global validity mask construction
    if args.vegetationmask and path.isfile(args.vegetationmask):
        key_vegmask = eoscale_manager.open_raster(
            raster_path=args.vegetationmask
        )
        key_valid_stack = eoexe.n_images_to_m_images_filter(
            inputs=[key_valid_stack, key_vegmask],
            image_filter=apply_vegetationmask,
            filter_parameters=vars(args),
            generate_output_profiles=eo_utils.single_bool_profile,
            stable_margin=0,
            context_manager=eoscale_manager,
            multiproc_context=args.multiproc_context,
            filter_desc="Valid stack processing with vegetationmask...",
        )
        key_valid_stack = key_valid_stack[0]
    if args.watermask and path.isfile(args.watermask):
        key_watermask = eoscale_manager.open_raster(
            raster_path=args.watermask
        )
        key_valid_stack = eoexe.n_images_to_m_images_filter(
            inputs=[key_valid_stack, key_watermask],
            image_filter=apply_watermask,
            filter_parameters={},
            generate_output_profiles=eo_utils.single_bool_profile,
            stable_margin=0,
            context_manager=eoscale_manager,
            multiproc_context=args.multiproc_context,
            filter_desc="Valid stack processing with watermask...",
        )
        key_valid_stack = key_valid_stack[0]
    # NDXI
    key_ndvi = eoscale_manager.open_raster(raster_path=args.file_ndvi)
    key_ndwi = eoscale_manager.open_raster(raster_path=args.file_ndwi)
    # WSF
    gt_key = eoscale_manager.open_raster(raster_path=args.extracted_wsf)
    return gt_key, key_ndvi, key_ndwi, key_original_valid_stack, key_phr, key_valid_stack


def samples_train_and_predict(args, eoscale_manager, key_ndvi, key_ndwi, key_original_valid_stack, key_phr,
                             key_valid_stack, keys_files_layers, x_samples, y_samples):
    """
    Trains a Random Forest classifier on provided samples and predicts an urban mask.
    Then, the predicted urban mask is saved to `args.urbanmask`.
    If `args.save_mode == "debug"`, also saves raw prediction probabilities.
    The function is called when there are sufficient building and non-building samples.
    """
    classifier = RandomForestClassifier(
        n_estimators=args.nb_estimators,
        max_depth=args.max_depth,
        class_weight="balanced",
        random_state=0,
        n_jobs=args.n_jobs,
    )
    logger.debug(
        "RandomForest parameters: \n%s\n",
        str(classifier.get_params())
    )
    random_forest_utils.train_classifier(classifier, x_samples, y_samples)
    random_forest_utils.print_feature_importance(
        classifier, args.files_layers
    )
    gc.collect()
    # Predict
    input_for_prediction = [
                               key_original_valid_stack,
                               key_valid_stack,
                               key_phr,
                               key_ndvi,
                               key_ndwi,
                           ] + keys_files_layers
    key_predict = eoexe.n_images_to_m_images_filter(
        inputs=input_for_prediction,
        image_filter=rf_prediction,
        filter_parameters={"classifier": classifier},
        generate_output_profiles=eo_utils.double_uint8_profile,
        stable_margin=0,
        context_manager=eoscale_manager,
        multiproc_context="spawn",
        filter_desc="RF prediction processing...",
    )
    time_random_forest = time.time()
    eoscale_manager.write(
        key=key_predict[0], img_path=args.urbanmask
    )  # classif
    if args.save_mode == "debug":
        eoscale_manager.write(
            key=key_predict[1],
            img_path=args.urbanmask.replace(
                ".tif", "_raw_predict.tif"
            ),
        )
    return time_random_forest

def add_nodata(
    input_buffer: list, input_profiles: list, params: dict
) -> np.ndarray:
    """
    Add nodata to the predicted mask

    :param list input_buffer: [original_valid_stack (without water and veg masks)]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: updated predicted mask (proba)
    """
    nodata_mask = (1 - input_buffer[0]).astype(bool)
    proba_buildings = np.full(nodata_mask.shape, params["fill_value"])
    proba_buildings[nodata_mask] = NODATA_INT8

    return proba_buildings


def getarguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Compute Urban Mask.")

    parser.add_argument(
        "main_config", help="First JSON file, load basis arguments"
    )
    parser.add_argument("-log_f",
                        "--logs_to_file",
                        action="store_true",
                        help="Store all logs to a file, instead of stdout",
                        )
    parser.add_argument("-d", "--debug", default=None, action="store_true", help="Debug flag")
    

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
        "-wsf",
        dest="extracted_wsf",
        help="Extracted World Settlement Footprint raster filename",
    )
    group1.add_argument(
        "-layers",
        nargs="+",
        dest="files_layers",
        help="Add layers as additional features used by learning algorithm",
    )
    group1.add_argument(
        "-watermask",
        help="Watermask filename (facultative) : "
        "urban mask will be learned & predicted, excepted on water areas",
    )
    group1.add_argument(
        "-vegetationmask",
        help="Vegetation mask filename (facultative) : "
        "urban mask will be learned & predicted, excepted on vegetated areas",
    )
    group1.add_argument(
        "-shadowmask",
        help="Shadowmask filename (facultative) : "
        "big shadow areas will be marked as background",
    )

    group2 = parser.add_argument_group(description="*** OPTIONS ***")
    group2.add_argument(
        "-vegmask_min_value",
        type=int,
        help="Vegetation min value for vegetated areas : all pixels with lower value will be predicted",
    )
    group2.add_argument(
        "-veg_binary_dilation",
        type=int,
        help="Size of disk structuring element (dilate non vegetated areas)",
    )
    group2.add_argument(
        "-value_classif",
        type=int,
        help="Input ground truth class to consider in the input ground truth",
    )
    group2.add_argument(
        "-gt_binary_erosion",
        type=int,
        action="store",
        help="Size of disk structuring element (erode GT before picking-up samples)",
    )
    group2.add_argument(
        "-save",
        choices=["none", "debug"],
        dest="save_mode",
        help="Save all files (debug) or only output mask (none)",
    )

    group3 = parser.add_argument_group(
        description="*** LEARNING SAMPLES SELECTION AND CLASSIFIER ***"
    )
    group3.add_argument(
        "-nb_samples_urban",
        type=int,
        help="Number of samples in buildings for learning",
    )
    group3.add_argument(
        "-nb_samples_other",
        type=int,
        help="Number of samples in other for learning",
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

    group4 = parser.add_argument_group(description="*** OUTPUT FILE ***")
    group4.add_argument("-urbanmask", help="Output classification filename")

    group5 = parser.add_argument_group(description="*** PARALLEL COMPUTING ***")
    group5.add_argument("-n_workers", type=int, help="Number of CPU")
    group5.add_argument(
        "-tile_max_size",
        type=int,
        help="Max tile size to be processed (0 : default)",
    )
    group5.add_argument(
        "-multiproc_context",
        default="spawn",
        help="Multiprocessing strategy: 'fork' or 'spawn' for EOScale",
    )
    args = parser.parse_args()

    arglist = []
    for arg in parser._actions:
        if arg.dest not in ["help"]:
            arglist.append(arg.dest)

    with open("args_list.json", 'w') as f:
        json.dump(arglist, f)

    return vars(args)


def slurp_urbanmask(main_config: str, logs_to_file: bool, debug: bool, user_config: str, file_vhr: str,
                    valid_stack: bool, file_ndvi: str, file_ndwi: str, extracted_wsf: str, files_layers: list, watermask: str,
                    vegetationmask: str, shadowmask: str, vegmask_min_value: int, veg_binary_dilation: int, value_classif: int,
                    gt_binary_erosion: int, save_mode: str, nb_samples_urban: int, nb_samples_other: int, max_depth: int,
                    nb_estimators: int, n_jobs: int, urbanmask: str, n_workers: int, tile_max_size: int, multiproc_context: str):
    """
    Main API to compute urban mask.
    """
    # Read the JSON files
    keys = [
        "input",
        "aux_layers",
        "masks",
        "resources",
        "post_process",
        "urban",
    ]
    argsdict, cli_params = utils.parse_args(keys, logs_to_file, main_config)

    for param in cli_params:
        # If the parameter from the CLI is not None, we update argsdict with the value from the CLI
        if locals()[param] is not None:
            argsdict[param] = locals()[param]

    logger.info("--" * 50)
    logger.info("SLURP - Urban mask\n")
    logger.info(f"JSON data loaded: {main_config}")
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

            # Build stack with all layers #
            gt_key, key_ndvi, key_ndwi, key_original_valid_stack, key_phr, key_valid_stack = build_stack_urban(args,
                                                                                                         eoscale_manager)

            time_stack = time.time()

            # BUILD SAMPLES

            # Recover useful features
            valid_stack = eoscale_manager.get_array(key_valid_stack)
            local_gt = eoscale_manager.get_array(gt_key)
            keys_files_layers = [
                eoscale_manager.open_raster(raster_path=args.files_layers[i])
                for i in range(len(args.files_layers))
            ]

            # Calculation of valid pixels
            nb_valid_pixels = np.count_nonzero(valid_stack)
            args.nb_valid_built_pixels = np.count_nonzero(
                np.logical_and(local_gt, valid_stack)
            )
            args.nb_valid_other_pixels = (
                nb_valid_pixels - args.nb_valid_built_pixels
            )

            building_areas = False
            non_building_areas = False

            if (
                args.nb_valid_built_pixels > args.nb_samples_urban
                and args.nb_valid_other_pixels > args.nb_samples_other
            ):
                # Nominal case : Ground Truth contains some pixels marked as building.
                nominal_case_urbanmask(args, eoscale_manager, gt_key, key_ndvi, key_ndwi, key_original_valid_stack,
                                       key_phr, key_valid_stack, keys_files_layers, t0, time_stack)

            elif args.nb_valid_built_pixels == nb_valid_pixels:
                # Corner case : no "non building pixels"
                logger.info(
                    f"**** Only urban areas in {args.file_vhr} -> mask saved as {args.urbanmask} ****"
                )

                key_predict = eoexe.n_images_to_m_images_filter(
                    inputs=[key_original_valid_stack],
                    image_filter=add_nodata,
                    filter_parameters={"fill_value": 100},
                    generate_output_profiles=eo_utils.single_uint8_profile,
                    context_manager=eoscale_manager,
                    multiproc_context=args.multiproc_context,
                    filter_desc="Add nodata...",
                )

                # Save proba mask
                eoscale_manager.write(
                    key=key_predict[0], img_path=args.urbanmask
                )
            else:
                # Corner case : no "building pixels" --> void mask (0)
                logger.info(
                    f"**** No urban areas in {args.file_vhr} -> void mask saved as {args.urbanmask} ****"
                )

                key_predict = eoexe.n_images_to_m_images_filter(
                    inputs=[key_original_valid_stack],
                    image_filter=add_nodata,
                    filter_parameters={"fill_value": 0},
                    generate_output_profiles=eo_utils.single_uint8_profile,
                    context_manager=eoscale_manager,
                    multiproc_context=args.multiproc_context,
                    filter_desc="Add nodata...",
                )

                # Save proba mask
                eoscale_manager.write(
                    key=key_predict[0], img_path=args.urbanmask
                )

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
    Main function to run the urban mask computation.
    It parses the command line arguments and calls the slurp_urbanmask function.
    """
    args = getarguments()
    slurp_urbanmask(**args)

if __name__ == "__main__":
    main()
