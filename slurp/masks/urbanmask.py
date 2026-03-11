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
import json
import logging
import time
import traceback
from os import path
from copy import deepcopy
from typing import List, Optional

import eoscale.eo_executors as eoexe
import eoscale.manager as eom
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from slurp.eomultiprocessing.slurp_executor import mp_n_to_m_images, mp_n_to_m_scalars
from slurp.eomultiprocessing.slurp_manager import slurpContextManager
from slurp.eomultiprocessing.utils import read_and_get_profile, write, read

from slurp.post_process.morphology import apply_morpho
from slurp.tools import profile_utils as eo_utils
from slurp.tools import random_forest_utils, utils
from slurp.tools.constant import NODATA_INT8

logger = logging.getLogger("slurp")

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ModuleNotFoundError:
    logger.error("Intel(R) Extension/Optimization for scikit-learn not found.")


def apply_vegetationmask(
    valid_stack: np.ndarray, vegmask: np.ndarray, vegmask_min_value: int, veg_binary_dilation: int
) -> np.ndarray:
    """
    Calculation of the valid pixels of a given image outside vegetation mask

    :param list input_buffer: VHR input image [valid_stack, vegetationmask]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments, must contain the keys
    "vegmask_min_value" and "veg_binary_dilation"
    :returns: valid_phr (boolean numpy array, True = valid data, False = no data)
    """
    non_veg = np.where(
        vegmask < vegmask_min_value, True, False
    )
    # dilate non vegetation areas, because sometimes the vegetation mask can cover urban areas
    non_veg_dilated = apply_morpho(
        non_veg, "binary_dilation", veg_binary_dilation
    )
    valid_stack = np.logical_and(valid_stack == 0, [non_veg_dilated])
    return valid_stack[0]


def apply_watermask(
    valid_stack: np.ndarray, watermask: np.ndarray
) -> np.ndarray:
    """
    Calculation of the valid pixels of a given image outside water mask

    :param list input_buffer: VHR input image [valid_stack, watermask]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments (not used but necessary for eoscale)
    :returns: valid_phr (boolean numpy array, True = valid data, False = no data)
    """
    valid_stack = np.logical_and(valid_stack == 0, watermask == 0)

    return valid_stack


def get_grid_indexes_from_mask(nb_samples, valid_mask, mask_ground_truth):
    """
    Recover of row and columns indices selected on the valid pixel of the image

    :param int nb_samples:
    :param boolean numpy array valid_mask :  shape (height, width)
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
        indices = np.arange(0, len(rows) - 1, int(len(rows) / nb_samples))
        s_rows = rows[indices]
        s_cols = cols[indices]
    else:
        s_rows = []
        s_cols = []

    return s_rows, s_cols


def build_samples(
    valid_stack: np.ndarray,
    gt: np.ndarray,
    phr1: np.ndarray,
    phr2: np.ndarray,
    phr3: np.ndarray,
    phr4: np.ndarray,
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    value_classif: int,
    gt_binary_erosion: int,
    nb_valid_built_pixels: int,
    nb_valid_other_pixels: int,
    nb_samples_urban: int,
    nb_samples_other: int,
    aux_inputs: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """
    Build training samples for urban detection.
    Algorithmically equivalent to eoscale implementation.
    """

    if aux_inputs is None:
        aux_inputs = []

    # ----------------------------------
    # VALIDITY MASK (same logic)
    # ----------------------------------

    validity_mask = (valid_stack == 0)

    # ----------------------------------
    # BUILDING MASK
    # ----------------------------------

    mask_building_before_erosion = np.where(
        gt == value_classif, True, False
    )

    mask_building = apply_morpho(
        mask_building_before_erosion,
        "binary_erosion",
        gt_binary_erosion,
    )

    # ----------------------------------
    # NON BUILDING MASK
    # ----------------------------------

    mask_non_building = np.where(gt == 0, True, False)

    # ----------------------------------
    # COUNT PIXELS (with validity AND)
    # ----------------------------------

    nb_built_subset = np.count_nonzero(
        np.logical_and(mask_building, validity_mask)
    )

    nb_other_subset = np.count_nonzero(
        np.logical_and(mask_non_building, validity_mask)
    )

    logger.debug(f"{nb_built_subset=} {nb_other_subset=}")

    # ----------------------------------
    # SAMPLING RATIOS
    # ----------------------------------

    urban_ratio = nb_built_subset / nb_valid_built_pixels
    other_ratio = nb_other_subset / nb_valid_other_pixels

    nb_urban_subsamples = round(urban_ratio * nb_samples_urban)
    nb_other_subsamples = round(other_ratio * nb_samples_other)

    # ----------------------------------
    # SAMPLE INDEX SELECTION
    # (same conditional logic)
    # ----------------------------------

    rows_b, cols_b = [], []
    rows_nb, cols_nb = [], []

    if nb_urban_subsamples > 0:
        rows_b, cols_b = get_grid_indexes_from_mask(
            nb_urban_subsamples,
            validity_mask,
            mask_building,
        )

        if nb_other_subsamples > 0:
            rows_nb, cols_nb = get_grid_indexes_from_mask(
                nb_other_subsamples,
                validity_mask,
                mask_non_building,
            )
    else:
        if nb_other_subsamples > 0:
            rows_nb, cols_nb = get_grid_indexes_from_mask(
                nb_other_subsamples,
                validity_mask,
                mask_non_building,
            )

    # ----------------------------------
    # MERGE INDICES
    # ----------------------------------

    rows = np.concatenate((rows_b, rows_nb))
    cols = np.concatenate((cols_b, cols_nb))

    # ----------------------------------
    # FEATURE STACK
    # Equivalent to concatenate(input_buffer[1:])
    # ----------------------------------

    features = [
        gt,
        phr1,
        phr2,
        phr3,
        phr4,
        ndvi,
        ndwi,
    ] + list(aux_inputs)

    im_stack = np.stack(features, axis=0)

    # ----------------------------------
    # SAMPLE PIXELS
    # ----------------------------------

    samples = np.transpose(
        im_stack[:, rows.astype(np.uint16), cols.astype(np.uint16)]
    )

    return samples


def rf_prediction(
    original_valid_stack: np.ndarray,
    current_valid_stack: np.ndarray,
    phr1: np.ndarray,
    phr2: np.ndarray,
    phr3: np.ndarray,
    phr4: np.ndarray,
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    aux_inputs=None,
    classifier=None,
    debug: bool = False,
) -> list:
    """
    Random Forest prediction for urban mask.
    Strict equivalent of legacy eoscale RF implementation.
    """

    # -------------------------------------------------
    # AUX INPUT NORMALIZATION
    # -------------------------------------------------
    if aux_inputs is None:
        aux_inputs = []

    if isinstance(aux_inputs, np.ndarray):
        aux_inputs = [aux_inputs]

    # -------------------------------------------------
    # REBUILD input_buffer SEMANTICS (algo reference)
    # -------------------------------------------------
    # In legacy algo:
    # input_buffer =
    # [original_valid_stack, valid_stack, vhr_image, ndvi, ndwi] + aux

    # VHR stack equivalent to input_buffer[2:]
    feature_layers = [
        phr1,
        phr2,
        phr3,
        phr4,
        ndvi,
        ndwi,
    ] + aux_inputs

    # concatenate exactly like legacy algo
    im_stack = np.concatenate(
        [layer[np.newaxis, ...] for layer in feature_layers],
        axis=0,
    )

    # -------------------------------------------------
    # MASKS (STRICT LEGACY LOGIC)
    # -------------------------------------------------
    nodata_mask = original_valid_stack != 0
    valid_mask = current_valid_stack == 0

    # legacy indexing behaviour
    buffer_to_predict = np.transpose(im_stack[:, valid_mask])

    if debug:
        logger.debug(
            "RF prediction buffer shape: %s",
            buffer_to_predict.shape,
        )

    # -------------------------------------------------
    # EMPTY TILE CORNER CASE
    # -------------------------------------------------
    if buffer_to_predict.shape[0] == 0:

        shape = valid_mask.shape
        prediction = np.full(shape, NODATA_INT8)
        proba_buildings = np.full(shape, NODATA_INT8)

        return proba_buildings, prediction

    # -------------------------------------------------
    # RANDOM FOREST PREDICTION
    # -------------------------------------------------
    proba = classifier.predict_proba(buffer_to_predict)

    res_classif = classifier.classes_.take(
        np.argmax(proba, axis=1),
        axis=0,
    )

    # normalize label convention
    res_classif[res_classif == 255] = 1

    # -------------------------------------------------
    # REBUILD IMAGE SPACE (LEGACY EXACT)
    # -------------------------------------------------
    prediction = np.zeros(valid_mask.shape)
    prediction[valid_mask] = res_classif
    prediction[nodata_mask] = NODATA_INT8

    proba_buildings = np.zeros(valid_mask.shape)
    proba_buildings[valid_mask] = 100 * proba[:, 1]
    proba_buildings[nodata_mask] = NODATA_INT8

    return proba_buildings, prediction


def nominal_case_urbanmask(
    args,
    slurp_manager,
    gt,
    ndvi,
    ndwi,
    phr,
    valid_stack,
    original_valid_stack,
    files_layers,
    margin,
    t0,
    time_stack,
    phr_profile,
    valid_stack_profile
):
    """
    Perform supervised classification to predict an urban mask using Random Forest.

    Pipeline
    --------
    1. Read auxiliary layers
    2. Build samples using multiprocessing
    3. Train Random Forest classifier
    4. Predict urban mask for full image
    """

    # ----------------------------------------------------
    # AUXILIARY LAYERS
    # ----------------------------------------------------

    if files_layers is None:
        files_layers = []
        keys_files_layers = []
    else:
        keys_files_layers = [
            band
            for path in files_layers
            for band in read(path)
        ]

    input_profile = deepcopy(valid_stack_profile)
    output_profile = eo_utils.double_uint8_profile(
        [deepcopy(valid_stack_profile)]
    )

    # ----------------------------------------------------
    # SAMPLE BUILDING
    # ----------------------------------------------------

    input_for_samples = [
        valid_stack[0][0],
        gt[0][0],
        phr[0][0],
        phr[0][1],
        phr[0][2],
        phr[0][3],
        ndvi[0][0],
        ndwi[0][0],
    ]

    samples = mp_n_to_m_scalars(
        inputs=input_for_samples,
        aux_inputs=keys_files_layers,
        image_height=input_profile["height"],
        image_width=input_profile["width"],
        context_manager=slurp_manager,
        func=build_samples,
        func_parameters={
            "value_classif": args.value_classif,
            "gt_binary_erosion": args.gt_binary_erosion,
            "nb_valid_other_pixels": args.nb_valid_other_pixels,
            "nb_valid_built_pixels": args.nb_valid_built_pixels,
            "nb_samples_urban": args.nb_samples_urban,
            "nb_samples_other": args.nb_samples_other,
        },
        reducer=utils.concatenate_samples,
    )

    logger.debug("samples:\n%s\n", samples)

    time_samples = time.time()

    # ----------------------------------------------------
    # SPLIT FEATURES / LABELS
    # ----------------------------------------------------

    x_samples = samples[:, 1:]
    y_samples = samples[:, 0]

    building_areas = len(np.where(y_samples == args.value_classif)[0]) > 0
    non_building_areas = len(np.where(y_samples == 0)[0]) > 0

    if not (building_areas and non_building_areas):

        logger.info(
            f"**** Corner case with too few urban samples for {args.file_vhr}"
        )
        output_profile = eo_utils.single_uint8_profile(
            [deepcopy(valid_stack_profile)]
        )
        predict = mp_n_to_m_images(
            inputs=[original_valid_stack[0][0]],
            image_height=input_profile["height"],
            image_width=input_profile["width"],
            output_profiles=[output_profile[0]],
            output_keys=["void_mask"],
            func=fill_constant_mask,
            func_parameters={"fill_value": 0},
            context_manager=slurp_manager,
            stable_margin=margin,
            binary=True,
        )

        return predict, output_profile, None, time_samples

    # ----------------------------------------------------
    # TRAIN RANDOM FOREST
    # ----------------------------------------------------

    classifier = RandomForestClassifier(
        n_estimators=args.nb_estimators,
        max_depth=args.max_depth,
        class_weight="balanced",
        random_state=0,
        n_jobs=args.n_jobs,
    )

    logger.debug(
        "RandomForest parameters:\n%s\n",
        str(classifier.get_params())
    )

    random_forest_utils.train_classifier(classifier, x_samples, y_samples)

    random_forest_utils.print_feature_importance(classifier, files_layers)

    gc.collect()

    utils.display_mem_usage(args.debug, "After training step")

    # ----------------------------------------------------
    # PREDICTION
    # ----------------------------------------------------

    input_for_prediction = [
        original_valid_stack[0][0],
        valid_stack[0][0],
        phr[0][0],
        phr[0][1],
        phr[0][2],
        phr[0][3],
        ndvi[0][0],
        ndwi[0][0]
    ]

    proba_buildings, predict  = mp_n_to_m_images(
        inputs=input_for_prediction,
        aux_inputs=keys_files_layers,
        image_height=input_profile["height"],
        image_width=input_profile["width"],
        output_profiles=[output_profile[0], output_profile[1]],
        output_keys=["proba_buildings", "prediction"],
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

    end_time = time.time()

    display_logs_rf(
        args,
        end_time,
        t0,
        time_samples,
        time_stack,
        time_random_forest,
    )

    return proba_buildings, output_profile[0], time_random_forest, time_samples


def display_logs_rf(
    args, end_time, t0, time_samples, time_stack, time_random_forest
):
    """
    Logs timing information and output paths for the urban mask generation pipeline.
    """
    logger.info(
        f"**** Urban proba mask for {args.file_vhr} (saved as {args.urbanmask}) ****"
    )
    logger.info(
        "Total time (user)       :\t" + utils.convert_time(end_time - t0)
    )
    logger.info(
        "- Build_stack           :\t" + utils.convert_time(time_stack - t0)
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


def build_stack_urban(args, slurp_manager):
    """
    Prepares and returns the required image layers and masks
    for urban mask processing using slurpContextManager.

    Parameters
    ----------
    args : Namespace
        Object containing paths and configuration parameters.

    slurp_manager : slurpContextManager
        SLURP context manager handling shared memory and raster access.

    Returns
    -------
    gt : list
    ndvi : list
    ndwi : list
    phr : list
    valid_stack : list
    margin : int
    phr_profile : dict
    """

    logger.info("Loading base rasters")

    # ==============================
    # PHR (VHR image)
    # ==============================

    key_phr, phr_profile = read_and_get_profile(args.file_vhr)

    args.nodata_phr = phr_profile.get("nodata")
    args.shape = (phr_profile["height"], phr_profile["width"])
    args.crs = phr_profile["crs"]
    args.transform = phr_profile["transform"]
    args.rpc = None

    # ==============================
    # VALID STACK
    # ==============================

    original_valid_stack, valid_stack_profile = read_and_get_profile(args.valid_stack)

    # ==============================
    # NDVI / NDWI
    # ==============================

    key_ndvi = read(args.file_ndvi)
    key_ndwi = read(args.file_ndwi)

    # ==============================
    # GROUND TRUTH (WSF)
    # ==============================

    key_gt = read(args.extracted_wsf)

    # ==============================
    # MARGIN
    # ==============================

    margin = 0

    # ==============================
    # APPLY VEGETATION MASK
    # ==============================

    if args.vegetationmask and path.isfile(args.vegetationmask):

        logger.info("Applying vegetation mask")

        vegmask = read(args.vegetationmask)

        input_profile = deepcopy(valid_stack_profile)
        output_profile = eo_utils.single_bool_profile([deepcopy(valid_stack_profile)])
        valid_stack = mp_n_to_m_images(
            inputs=[
                original_valid_stack[0],
                vegmask[0],
            ],
            image_height=input_profile["height"],
            image_width=input_profile["width"],
            output_profiles=[output_profile],
            output_keys=["valid_stack_veg"],
            func=apply_vegetationmask,
            func_parameters={
                "vegmask_min_value": args.vegmask_min_value,
                "veg_binary_dilation": args.veg_binary_dilation
            },
            context_manager=slurp_manager,
            stable_margin=margin,
            binary=True,
        )

    # ==============================
    # APPLY WATER MASK
    # ==============================

    if args.watermask and path.isfile(args.watermask):

        logger.info("Applying water mask")

        watermask = read(args.watermask)

        input_profile = deepcopy(valid_stack_profile)
        output_profile = eo_utils.single_bool_profile([deepcopy(valid_stack_profile)])

        valid_stack = mp_n_to_m_images(
            inputs=[
                valid_stack[0],
                watermask[0],
            ],
            image_height=input_profile["height"],
            image_width=input_profile["width"],
            output_profiles=[output_profile],
            output_keys=["valid_stack_water"],
            func=apply_watermask,
            func_parameters={},
            context_manager=slurp_manager,
            stable_margin=margin,
            binary=True,
        )

    logger.info("Urban stack ready")

    return (
        [key_gt],
        [key_ndvi],
        [key_ndwi],
        [key_phr],
        [valid_stack],
        [original_valid_stack],
        margin,
        phr_profile,
        valid_stack_profile
    )


def samples_train_and_predict(
    args,
    eoscale_manager,
    key_ndvi,
    key_ndwi,
    key_original_valid_stack,
    key_phr,
    key_valid_stack,
    keys_files_layers,
    x_samples,
    y_samples,
):
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
        "RandomForest parameters: \n%s\n", str(classifier.get_params())
    )
    random_forest_utils.train_classifier(classifier, x_samples, y_samples)
    random_forest_utils.print_feature_importance(classifier, args.files_layers)
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
            img_path=args.urbanmask.replace(".tif", "_raw_predict.tif"),
        )
    return time_random_forest


def fill_constant_mask(
    valid_stack: np.ndarray, fill_value: int
) -> np.ndarray:
    """
    Add nodata to the predicted mask

    :param list input_buffer: [original_valid_stack (without water and veg masks)]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: updated predicted mask (proba)
    """

    proba_buildings = np.where(
        valid_stack == 0, fill_value, NODATA_INT8
    )

    return proba_buildings


def getarguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Compute Urban Mask.")

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
        "-d", "--debug", default=None, action="store_true", help="Debug flag"
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
        help=(
            "Vegetation min value for vegetated areas: all pixels with lower value "
            "will be predicted"
        ),
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

    with open("args_list.json", "w") as f:
        json.dump(arglist, f)

    return vars(args)


def slurp_urbanmask(
    main_config: str,
    logs_to_file: bool,
    debug: bool,
    user_config: str,
    file_vhr: str,
    valid_stack: bool,
    file_ndvi: str,
    file_ndwi: str,
    extracted_wsf: str,
    files_layers: list,
    watermask: str,
    vegetationmask: str,
    shadowmask: str,
    vegmask_min_value: int,
    veg_binary_dilation: int,
    value_classif: int,
    gt_binary_erosion: int,
    save_mode: str,
    nb_samples_urban: int,
    nb_samples_other: int,
    max_depth: int,
    nb_estimators: int,
    n_jobs: int,
    urbanmask: str,
    n_workers: int,
    tile_max_size: int,
    multiproc_context: str,
):
    """
    Main API to compute urban mask using slurpContextManager
    """

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
        if locals()[param] is not None:
            argsdict[param] = locals()[param]

    logger.info("--" * 50)
    logger.info("SLURP - Urban mask\n")
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
            # BUILD STACK
            # ==============================

            logger.info("[0] Step: Build stack")

            (
                gt,
                ndvi,
                ndwi,
                phr,
                valid_stack,
                original_valid_stack,
                margin,
                phr_profile,
                valid_stack_profile
            ) = build_stack_urban(args, slurp_manager)

            time_stack = time.time()

            # ==============================
            # BUILD SAMPLES
            # ==============================

            logger.info("[1] Step: Build samples")

            local_gt = gt[0][0]
            local_valid = valid_stack[0][0]

            nb_valid_pixels = np.sum(local_valid == 0)

            args.nb_valid_built_pixels = np.count_nonzero(
                np.logical_and(local_gt, local_valid == 0)
            )

            args.nb_valid_other_pixels = (
                nb_valid_pixels - args.nb_valid_built_pixels
            )

            # ==============================
            # NOMINAL CASE
            # ==============================

            if (
                args.nb_valid_built_pixels > args.nb_samples_urban
                and args.nb_valid_other_pixels > args.nb_samples_other
            ):

                logger.info("[2] Step: RF URBAN")

                predict, output_profile, time_random_forest, time_samples = nominal_case_urbanmask(
                    args,
                    slurp_manager,
                    gt,
                    ndvi,
                    ndwi,
                    phr,
                    valid_stack,
                    original_valid_stack,
                    files_layers,
                    margin,
                    t0,
                    time_stack,
                    phr_profile,
                    valid_stack_profile
                )
                slurp_manager.write_tif(
                    data=predict,
                    path=args.urbanmask,
                    target_profile=output_profile
                )


            # ==============================
            # ALL PIXELS = URBAN
            # ==============================

            elif args.nb_valid_built_pixels == nb_valid_pixels:

                logger.info(
                    f"**** Only urban areas in {args.file_vhr} "
                    f"-> mask saved as {args.urbanmask} ****"
                )

                input_profile = deepcopy(phr_profile)
                output_profile = eo_utils.single_uint8_profile(
                    [deepcopy(phr_profile)]
                )

                predict = mp_n_to_m_images(
                    inputs=[valid_stack[0][0]],
                    image_height=input_profile["height"],
                    image_width=input_profile["width"],
                    output_profiles=[output_profile],
                    output_keys=[path.basename(args.urbanmask)],
                    func=fill_constant_mask,
                    func_parameters={"fill_value": 100},
                    context_manager=slurp_manager,
                    stable_margin=margin,
                    binary=True,
                )

            # ==============================
            # NO URBAN PIXELS
            # ==============================

            else:

                logger.info(
                    f"**** No urban areas in {args.file_vhr} "
                    f"-> void mask saved as {args.urbanmask} ****"
                )

                input_profile = deepcopy(phr_profile)
                output_profile = eo_utils.single_uint8_profile(
                    [deepcopy(phr_profile)]
                )

                predict = mp_n_to_m_images(
                    inputs=[valid_stack[0][0]],
                    image_height=input_profile["height"],
                    image_width=input_profile["width"],
                    output_profiles=[output_profile],
                    output_keys=[path.basename(args.urbanmask)],
                    func=fill_constant_mask,
                    func_parameters={"fill_value": 0},
                    context_manager=slurp_manager,
                    stable_margin=margin,
                    binary=True,
                )

            t1 = time.time()

            logger.info(
                "Total time (user):\t" + utils.convert_time(t1 - t0)
            )

        except Exception:
            logger.error("Unexpected error:", exc_info=True)
            traceback.print_exc()

    logger.info("End of urbanmask step\n")

def main():
    """
    Main function to run the urban mask computation.
    It parses the command line arguments and calls the slurp_urbanmask function.
    """
    args = getarguments()
    slurp_urbanmask(**args)


if __name__ == "__main__":
    main()
