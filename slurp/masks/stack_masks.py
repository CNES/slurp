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


"""
This script stacks existing masks

Final mask values
- 1st layer : class
- 2nd layer : estimation of elevation
"""

import argparse
import json
import logging
import time
import traceback
from os import path
from copy import deepcopy

import numpy as np
from skimage import segmentation
from skimage.filters import sobel
from skimage.measure import label

from slurp.eomultiprocessing.slurp_executor import mp_n_to_m_images, mp_n_to_m_scalars
from slurp.eomultiprocessing.slurp_manager import slurpContextManager
from slurp.eomultiprocessing.utils import read_and_get_profile, write, read
from slurp.post_process.morphology import apply_morpho, morpho_clean
from slurp.tools import profile_utils as eo_utils
from slurp.tools import io_utils, utils
from slurp.tools.constant import HIGH, LOW, NODATA_INT8

logger = logging.getLogger("slurp")

# Values from Copernicus WaterBodyMask
SEA = 1
LAKE = 2
RIVER = 3


def build_stack_stackmask(args, slurp_manager):
    """
    Prepare inputs required for stack mask post-processing.

    Parameters
    ----------
    args : argparse.Namespace
        Input arguments containing paths to required rasters.
    slurp_manager : object
        SLURP manager instance (unused but kept for interface consistency).

    Returns
    -------
    tuple
        (
            vhr_image_list,
            valid_stack_list,
            watermask_list,
            vegmask_list,
            urbanmask_list,
            shadowmask_list,
            wsf_list,
            margin,
            image_profile,
        )
    """

    logger.info("Loading base rasters")

    # ==============================
    # VHR IMAGE
    # ==============================

    key_image, image_profile = read_and_get_profile(args.file_vhr)
    input_profile = deepcopy(image_profile)

    args.shape = (input_profile["height"], input_profile["width"])
    args.crs = input_profile["crs"]
    args.transform = input_profile["transform"]
    args.nodata_vhr = 0

    # ==============================
    # REQUIRED MASKS
    # ==============================

    key_validstack = read(args.valid_stack)
    key_watermask = read(args.watermask)
    key_vegmask = read(args.vegetationmask)
    key_urbanmask = read(args.urbanmask)
    key_shadowmask = read(args.shadowmask)
    key_wsf = read(args.extracted_wsf)

    margin = 100  # identical to previous stable_margin

    logger.info("Stackmask inputs ready")

    return (
        [key_image],
        [key_validstack],
        [key_watermask],
        [key_vegmask],
        [key_urbanmask],
        [key_shadowmask],
        [key_wsf],
        margin,
        image_profile,
    )

def watershed_regul_buildings(
    input_image: np.ndarray,
    urbanmask: np.ndarray,
    wsf: np.ndarray,
    vegmask: np.ndarray,
    watermask: np.ndarray,
    shadowmask: np.ndarray,
    *,
    sobel_image: str,
    regul_type: str,
    building_threshold: int,
    building_erosion: int,
    erosion_radius: int,
    bonus_gt: int,
    malus_shadow: int,
    value_classif_bare_ground: int,
    value_classif_buildings: int,
    value_classif_low_veg: int,
    value_classif_high_veg: int,
    value_classif_false_positive_buildings: int,
    value_classif_background: int,
):
    """
    Apply watershed-based regularization to refine building classification.

    Parameters
    ----------
    input_image : np.ndarray
        Multi-band VHR image (bands, height, width).
    urbanmask : np.ndarray
        Urban probability or classification mask.
    wsf : np.ndarray
        World Settlement Footprint mask used as building ground truth.
    vegmask : np.ndarray
        Vegetation classification mask.
    watermask : np.ndarray
        Binary water mask.
    shadowmask : np.ndarray
        Shadow classification mask.
    sobel_image : str
        Image type used for edge detection ("ndvi", "nir", or "rgb").
    regul_type : str
        Regulation mode controlling which classes are enforced.
    building_threshold : int
        Percentile threshold applied to urbanmask.
    building_erosion : int
        Erosion radius applied to detected buildings.
    erosion_radius : int
        Morphological erosion radius for marker generation.
    bonus_gt : int
        Bonus value applied to ground-truth building pixels.
    malus_shadow : int
        Penalty applied to shadow pixels.
    value_classif_bare_ground : int
        Output value for bare ground class.
    value_classif_buildings : int
        Output value for buildings class.
    value_classif_low_veg : int
        Output value for low vegetation class.
    value_classif_high_veg : int
        Output value for high vegetation class.
    value_classif_false_positive_buildings : int
        Output value for false positive buildings.
    value_classif_background : int
        Output value for background class.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (segmentation_map, markers)
    """

    # Mono image for edge detection
    if sobel_image == "ndvi":
        im_mono = (input_image[3] - input_image[0]) / (input_image[3] + input_image[0])
    elif sobel_image == "nir":
        im_mono = input_image[3]
    else:
        im_mono = 0.29 * input_image[0] + 0.58 * input_image[1] + 0.114 * input_image[2]

    edges = sobel(im_mono)
    markers = np.zeros((1, input_image.shape[1], input_image.shape[2]))

    # Bare ground
    eroded_bare_ground = apply_morpho(vegmask == 11, "binary_erosion", erosion_radius)
    markers[0][eroded_bare_ground] = value_classif_bare_ground if regul_type == "all" else value_classif_background

    # Buildings
    ground_truth_eroded = apply_morpho(wsf == 255, "binary_erosion", erosion_radius)
    normalized_building_threshold = np.percentile(urbanmask, building_threshold)
    urbanmask[ground_truth_eroded] += bonus_gt
    urbanmask[shadowmask == 2] -= malus_shadow
    probable_buildings = np.logical_and(ground_truth_eroded, urbanmask > normalized_building_threshold)
    probable_buildings = apply_morpho(probable_buildings, "binary_erosion", building_erosion)
    false_positive = np.logical_and(apply_morpho(wsf == 255, "binary_dilation", 10) == 0, urbanmask > normalized_building_threshold)
    markers[0][probable_buildings] = value_classif_buildings
    markers[0][false_positive] = value_classif_false_positive_buildings

    # Low vegetation
    eroded_low_veg = apply_morpho(vegmask == 21, "binary_erosion", erosion_radius)
    markers[0][eroded_low_veg] = value_classif_low_veg if regul_type == "all" else value_classif_background

    # High vegetation
    eroded_high_veg = apply_morpho(np.logical_or(vegmask == 23, vegmask == 22), "binary_erosion", erosion_radius)
    markers[0][eroded_high_veg] = value_classif_high_veg if regul_type == "all" else value_classif_background

    # Shadow
    eroded_shadow = apply_morpho(shadowmask == 2, "binary_erosion", erosion_radius)
    markers[0][eroded_shadow] = value_classif_background

    # Water
    markers[0][watermask == 1] = value_classif_background

    seg = segmentation.watershed(edges, markers[0].astype(np.uint8))
    return seg, markers


def infer_waterbodies_type(wbm, watermask, params):
    """
    Clean and apply watershed regulation for the watermask

    :param np.ndarray wbm: WBM file from post_process function
    :param np.ndarray watermask: Watermask created by the dedicated script
    :param dict params: dictionary of arguments
    :return: categorized mask
    """
    nb_iter = 2
    clean_watermask = watermask == 1
    for _ in range(nb_iter):
        clean_watermask = apply_morpho(clean_watermask, "binary_opening", 2)

    # remove small objects in order to reduce the segmentation
    watermask_remove = apply_morpho(
        clean_watermask,
        "remove_small_objects",
        params["minimal_size_water_area"],
    )

    watermask_remove = apply_morpho(
        watermask_remove,
        "remove_small_holes",
        params["minimal_size_water_area"],
    )

    # 2nd step: segmentation
    # label image regions
    label_image = label(watermask_remove)
    logger.debug(
        f"Infer waterbodies type --> {len(np.unique(label_image))} water bodies found"
    )

    categorized_watermask = np.zeros_like(label_image)

    list_classes = {
        SEA: params["value_classif_sea"],
        LAKE: params["value_classif_lake"],
        RIVER: params["value_classif_river"],
        0: 0,  # params["value_classif_water"]
    }
    # loop on all water bodies
    val_uniques = np.unique(label_image)
    for val in val_uniques:
        mask_val = label_image == val
        # values from WBM covered by value 'val' in label_image
        if 1 in watermask_remove[mask_val]:
            # water body
            sub_set_wbm = wbm[mask_val]
            kind_waterbodies = np.unique(sub_set_wbm)
            # We select water body type by priority order
            # sea > lake > river > (other) water
            main_wb = choose_wb(kind_waterbodies)
            categorized_watermask[mask_val] = list_classes[main_wb]

    return categorized_watermask


def choose_wb(kind_waterbodies):
    if SEA in kind_waterbodies:
        return SEA
    elif LAKE in kind_waterbodies:
        return LAKE
    elif RIVER in kind_waterbodies:
        return RIVER
    else:
        return 0


def post_process(
    phr1: np.ndarray,
    phr2: np.ndarray,
    phr3: np.ndarray,
    phr4: np.ndarray,
    valid_stack: np.ndarray,
    watermask: np.ndarray,
    vegmask: np.ndarray,
    urbanmask: np.ndarray,
    shadowmask: np.ndarray,
    wsf: np.ndarray,
    *,
    winter_vegetation: bool,
    value_classif_bare_ground: int,
    value_classif_buildings: int,
    value_classif_water: int,
    value_classif_low_veg: int,
    value_classif_high_veg: int,
    value_classif_false_positive_buildings: int,
    value_classif_background: int,
    sobel_image: str,
    regul_type: str,
    building_threshold: int,
    building_erosion: int,
    erosion_radius: int,
    bonus_gt: int,
    malus_shadow: int,
    binary_closing: int | None = None,
    binary_opening: int | None = None,
    remove_small_holes: int | None = None,
    remove_small_objects: int | None = None,
):
    """
    Perform SLURP post-processing to generate final classification layers.

    Parameters
    ----------
    phr1, phr2, phr3, phr4 : np.ndarray
        Four spectral bands of the VHR image.
    valid_stack : np.ndarray
        Validity mask defining nodata regions.
    watermask : np.ndarray
        Binary water mask.
    vegmask : np.ndarray
        Vegetation classification mask.
    urbanmask : np.ndarray
        Urban probability or classification mask.
    shadowmask : np.ndarray
        Shadow classification mask.
    wsf : np.ndarray
        World Settlement Footprint mask.
    winter_vegetation : bool
        Enable winter vegetation correction.
    value_classif_bare_ground : int
        Output value for bare ground class.
    value_classif_buildings : int
        Output value for buildings class.
    value_classif_water : int
        Output value for water class.
    value_classif_low_veg : int
        Output value for low vegetation class.
    value_classif_high_veg : int
        Output value for high vegetation class.
    value_classif_false_positive_buildings : int
        Output value for false positive buildings.
    value_classif_background : int
        Output value for background class.
    sobel_image : str
        Image type used for watershed edge detection.
    regul_type : str
        Regulation mode applied during watershed segmentation.
    building_threshold : int
        Percentile threshold for building detection.
    building_erosion : int
        Morphological erosion radius for buildings.
    erosion_radius : int
        Radius used for marker erosion.
    bonus_gt : int
        Bonus applied to ground-truth buildings.
    malus_shadow : int
        Penalty applied to shadow areas.
    binary_closing : int | None, optional
        Closing kernel size for morphological cleaning.
    binary_opening : int | None, optional
        Opening kernel size for morphological cleaning.
    remove_small_holes : int | None, optional
        Minimum hole size removed during cleaning.
    remove_small_objects : int | None, optional
        Minimum object size removed during cleaning.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (classification_layer, height_layer, markers_layer)
    """

    # Combine bands into a single array for watershed input
    input_image = np.stack([phr1, phr2, phr3, phr4], axis=0)
    h, w = phr1.shape

    stack = np.zeros((1, h, w), dtype=np.uint8)

    # ==============================
    # Winter vegetation correction
    # ==============================
    if winter_vegetation:
        mix_textured_veg = np.logical_or(
            vegmask == 12,
            vegmask == 13,
        )
        vegmask[mix_textured_veg] = 22

    # ==============================
    # Watershed regularisation
    # ==============================
    seg, markers = watershed_regul_buildings(
        input_image=input_image,
        urbanmask=urbanmask,
        wsf=wsf,
        vegmask=vegmask,
        watermask=watermask,
        shadowmask=shadowmask,
        sobel_image=sobel_image,
        regul_type=regul_type,
        building_threshold=building_threshold,
        building_erosion=building_erosion,
        erosion_radius=erosion_radius,
        bonus_gt=bonus_gt,
        malus_shadow=malus_shadow,
        value_classif_bare_ground=value_classif_bare_ground,
        value_classif_buildings=value_classif_buildings,
        value_classif_low_veg=value_classif_low_veg,
        value_classif_high_veg=value_classif_high_veg,
        value_classif_false_positive_buildings=value_classif_false_positive_buildings,
        value_classif_background=value_classif_background,
    )
    logger.debug(f"{np.unique(seg)=}")

    # ==============================
    # Classes cleaning
    # ==============================
    clean_bare_ground = (
    morpho_clean(
        seg == value_classif_bare_ground,
        binary_closing=binary_closing,
        binary_opening=binary_opening,
        remove_small_holes=remove_small_holes,
        remove_small_objects=remove_small_objects,
    ) == 1
)
    stack[0][clean_bare_ground] = value_classif_bare_ground

    clean_buildings = morpho_clean(
        seg == value_classif_buildings, 
        binary_closing=binary_closing,
        binary_opening=binary_opening,
        remove_small_holes=remove_small_holes,
        remove_small_objects=remove_small_objects,
    ) == 1
    stack[0][clean_buildings] = value_classif_buildings

    stack[0][watermask == 1] = value_classif_water

    clean_low_veg = vegmask == 21
    stack[0][clean_low_veg] = value_classif_low_veg

    clean_high_veg = vegmask == 22
    stack[0][clean_high_veg] = value_classif_high_veg

    stack[0][valid_stack != 0] = NODATA_INT8

    # ==============================
    # HEIGHT LAYER
    # ==============================
    height_layer = np.zeros((1, h, w), dtype=np.uint8)
    height_layer[0][clean_bare_ground] = LOW
    height_layer[0][clean_low_veg] = LOW
    height_layer[0][clean_buildings] = HIGH
    height_layer[0][clean_high_veg] = HIGH
    height_layer[0][watermask == 1] = 0
    height_layer[0][shadowmask == 2] = 0
    height_layer[0][valid_stack != 0] = NODATA_INT8

    # ==============================
    # MARKERS
    # ==============================
    markers_layer = np.zeros((1, h, w), dtype=np.uint8)
    markers_layer[0] = markers
    markers_layer[0][valid_stack != 0] = NODATA_INT8

    return stack[0], height_layer[0], markers_layer[0]


def getarguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()

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
    group1.add_argument("-vegetationmask", help="Vegetation mask")
    group1.add_argument("-watermask", help="Water mask")
    group1.add_argument("-urbanmask", help="Urban mask probabilities")
    group1.add_argument("-shadowmask", help="Shadow mask")
    group1.add_argument(
        "-wsf",
        dest="extracted_wsf",
        help="Extracted World Settlement Footprint raster  filename",
    )
    group1.add_argument(
        "-wbm",
        dest="extracted_wbm",
        help="Extracted Water Body Mask raster filename",
    )

    group2 = parser.add_argument_group(description="*** WATERSHED OPTIONS ***")
    group2.add_argument(
        "-sobel_image",
        type=str,
        default="rgb",
        choices=["rgb", "nir", "ndvi"],
        help="Select layer on which compute sobel and perform watershed "
        "regularization (default : rgb)",
    )

    group2.add_argument(
        "-regul_type",
        type=str,
        default="all",
        choices=["all", "building"],
        help="Regularization type : all (building, vegetation, bare ground),"
        " building (building only)",
    )

    group2.add_argument(
        "-building_threshold",
        type=int,
        help="Threshold to consider building as detected",
    )
    group2.add_argument(
        "-building_erosion",
        type=int,
        help="Supposed buildings will be eroded by this radius in the marker step",
    )
    group2.add_argument(
        "-erosion_radius",
        type=int,
        help="Other classes than buildings will be eroded by this radius in the marker step",
    )
    group2.add_argument(
        "-bonus_gt",
        type=int,
        help="Bonus for pixels covered by GT, in the watershed regularization step "
        "(ex : +30 to improve discrimination between building and background)",
    )
    group2.add_argument(
        "-malus_shadow",
        type=int,
        help="Value of the malus for pixels in shadow, in the watershed regularization step",
    )

    group2.add_argument(
        "-winter_vegetation",
        action="store_true",
        help="Consider textured 'mixed vegetation areas' as high vegetation "
        "(to take into account trees with no leaves)",
    )

    group2.add_argument(
        "--categorized_watermask",
        dest="categorized_watermask",
        action="store_true",
        help="Compute a categorized watermask based on the WBM file",
    )
    group2.add_argument(
        "--no_categorized_watermask",
        dest="categorized_watermask",
        action="store_false",
        help="Don't compute the categorize watermask",
    )

    group2.add_argument(
        "-minimal_size_water_area",
        type=int,
        default=10000,
        help="Minimal area (in pixels) of water bodies",
    )

    group4 = parser.add_argument_group(description="*** OUTPUT FILE ***")
    group4.add_argument("-stackmask", help="Output Final mask filename")
    group4.add_argument(
        "-low_veg",
        dest="value_classif_low_veg",
        type=int,
        help="Output classification value for low vegetation",
    )
    group4.add_argument(
        "-high_veg",
        dest="value_classif_high_veg",
        type=int,
        help="Output classification value for high vegetation",
    )
    group4.add_argument(
        "-water",
        dest="value_classif_water",
        type=int,
        help="Output classification value for water",
    )
    group4.add_argument(
        "-buildings",
        dest="value_classif_buildings",
        type=int,
        help="Output classification value for buildings",
    )
    group4.add_argument(
        "-bare_ground",
        dest="value_classif_bare_ground",
        type=int,
        help="Output classification value for bare ground",
    )
    group4.add_argument(
        "-false_pos_buildings",
        dest="value_classif_false_positive_buildings",
        type=int,
        help="Output classification value for buildings false positive",
    )
    group4.add_argument(
        "-background",
        dest="value_classif_background",
        type=int,
        help="Output classification value for background",
    )

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

def slurp_stackmask(
    main_config: str,
    logs_to_file: bool,
    debug: bool,
    user_config: str,
    file_vhr: str,
    valid_stack: bool,
    vegetationmask: str,
    watermask: str,
    urbanmask: str,
    shadowmask: str,
    extracted_wsf: str,
    extracted_wbm: str,
    sobel_image: str,
    regul_type: str,
    building_threshold: int,
    building_erosion: int,
    erosion_radius: int,
    bonus_gt: int,
    winter_vegetation: bool,
    malus_shadow: int,
    stackmask: str,
    value_classif_low_veg: int,
    value_classif_high_veg: int,
    value_classif_water: int,
    value_classif_buildings: int,
    value_classif_bare_ground: int,
    value_classif_false_positive_buildings: int,
    value_classif_background: int,
    n_workers: int,
    tile_max_size: int,
    multiproc_context: str,
    categorized_watermask: bool,
    minimal_size_water_area: int,
):

    keys = [
        "input",
        "prepare",
        "aux_layers",
        "masks",
        "resources",
        "post_process",
        "stack",
    ]

    argsdict, cli_params = utils.parse_args(keys, logs_to_file, main_config)

    for param in cli_params:
        if locals()[param] is not None:
            argsdict[param] = locals()[param]

    logger.info("--" * 50)
    logger.info("SLURP - Stack masks\n")
    logger.info(f"JSON data loaded: {main_config}")

    args = argparse.Namespace(**argsdict)

    if args.debug:
        logger.handlers[0].setLevel(logging.DEBUG)

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
                image,
                validstack,
                watermask,
                vegmask,
                urbanmask,
                shadowmask,
                wsf,
                margin,
                image_profile,
            ) = build_stack_stackmask(args, slurp_manager)

            # ==============================
            # POST PROCESS
            # ==============================

            logger.info("[1] Step: Post processing")

            input_profile = deepcopy(image_profile)
            output_profile = eo_utils.three_uint8_profile(
                [deepcopy(image_profile)]
            )

            stack, height, markers  = mp_n_to_m_images(
                inputs=[
                    image[0][0],  # phr1
                    image[0][1],  # phr2
                    image[0][2],  # phr3
                    image[0][3],  # phr4
                    validstack[0][0],
                    watermask[0][0],
                    vegmask[0][0],
                    urbanmask[0][0],
                    shadowmask[0][0],
                    wsf[0][0],
                ],
                image_height=input_profile["height"],
                image_width=input_profile["width"],
                output_profiles=output_profile,
                output_keys=["stack", "height", "markers"],
                func=post_process,
                func_parameters=dict(
                    winter_vegetation=args.winter_vegetation,
                    value_classif_bare_ground=args.value_classif_bare_ground,
                    value_classif_buildings=args.value_classif_buildings,
                    value_classif_water=args.value_classif_water,
                    value_classif_low_veg=args.value_classif_low_veg,
                    value_classif_high_veg=args.value_classif_high_veg,
                    value_classif_false_positive_buildings=args.value_classif_false_positive_buildings,
                    value_classif_background=args.value_classif_background,
                    sobel_image=args.sobel_image,
                    regul_type=args.regul_type,
                    building_threshold=args.building_threshold,
                    building_erosion=args.building_erosion,
                    erosion_radius=args.erosion_radius,
                    bonus_gt=args.bonus_gt,
                    malus_shadow=args.malus_shadow,
                    binary_closing=args.binary_closing,
                    binary_opening=args.binary_opening,
                    remove_small_holes=args.remove_small_holes,
                    remove_small_objects=args.remove_small_objects,
                ),
                context_manager=slurp_manager,
                stable_margin=margin,
            )

            # ==============================
            # WRITE STACK
            # ==============================

            if args.categorized_watermask:

                key_wbm = read(args.extracted_wbm)
                wbm = key_wbm[0]
                water = watermask[0][0]

                categorized = infer_waterbodies_type(
                    wbm,
                    water,
                    vars(args),
                )

                stack[:] = np.where(categorized != 0, categorized, stack)
                io_utils.save_image(
                    stack,
                    args.stackmask,
                    crs=output_profile[0]["crs"],
                    transform=output_profile[0]["transform"],
                )
            else:
                slurp_manager.write_tif(
                    data=stack,
                    path=args.stackmask,
                    target_profile=output_profile[0],
                )

            if args.debug:

                base = path.splitext(args.stackmask)[0]

                slurp_manager.write_tif(
                    data=height,
                    path=base + "_height.tif",
                    target_profile=output_profile[1],
                )

                slurp_manager.write_tif(
                    data=markers,
                    path=base + "_markers.tif",
                    target_profile=output_profile[2],
                )

            t1 = time.time()

            logger.info(
                f"**** Stack masks saved as {args.stackmask} ****"
            )
            logger.info(
                "Total time (user):\t"
                + utils.convert_time(t1 - t0)
            )

        except Exception:
            logger.error("Unexpected error:", exc_info=True)
            traceback.print_exc()

    logger.info("End of stackmask step\n")


def main():
    """
    Main function to run the stack mask computation.
    It parses the command line arguments and calls the slurp_stackmask function.
    """
    args = getarguments()
    slurp_stackmask(**args)


if __name__ == "__main__":
    main()
