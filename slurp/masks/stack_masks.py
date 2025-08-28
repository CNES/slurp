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
import time
import traceback
import logging
import pathlib
import json
from os import makedirs, path, remove


import numpy as np
from skimage import segmentation
from skimage.filters import sobel
from skimage.util import map_array
from skimage.measure import label, regionprops

import eoscale.eo_executors as eoexe
import eoscale.manager as eom

from slurp.post_process.morphology import apply_morpho, morpho_clean
from slurp.tools import eoscale_utils as eo_utils
from slurp.tools import io_utils, utils
from slurp.tools.constant import NODATA_INT8, LOW, HIGH


logger = logging.getLogger("slurp")

def watershed_regul_buildings(
    input_image, urbanmask, wsf, vegmask, watermask, shadowmask, params
):
    """
    Clean and apply watershed regulation for buildings

    :param np.ndarray input_image: VHR input image
    :param np.ndarray urbanmask: Urbanmask created by the dedicated script
    :param np.ndarray wsf: WSF file from post_process function
    :param np.ndarray vegmask: Vegetationnmask created by the dedicated script
    :param np.ndarray watermask: Watermask created by the dedicated script
    :param np.ndarray shadowmask: Shadowmask created by the dedicated script
    :param dict params: dictionary of arguments
    :return: tuple of segmentation value and markers
    """
    # Compute mono image from RGB image
    # im_mono = 0.29*input_image[0] + 0.58*input_image[1] + 0.114*input_image[2]
    im_mono = 0.3 * input_image[0] + 0.3 * input_image[1] + 0.3 * input_image[3]
    edges = sobel(im_mono)

    markers = np.zeros((1, input_image.shape[1], input_image.shape[2]))

    # We set markers by reverse order of confidence
    eroded_bare_ground = apply_morpho(
        vegmask[0] == 11, "binary_erosion", params["building_erosion"]
    )
    markers[0][eroded_bare_ground] = params["value_classif_bare_ground"]

    ground_truth_eroded = apply_morpho(
        wsf[0] == 255, "binary_erosion", params["building_erosion"]
    )

    # Bonus for pixels above ground truth
    urbanmask[0][ground_truth_eroded] += params["bonus_gt"]
    # Malus for pixels in shadow areas
    urbanmask[0][shadowmask[0] == 2] -= params["malus_shadow"]
    probable_buildings = np.logical_and(
        ground_truth_eroded, urbanmask[0] > params["building_threshold"]
    )
    probable_buildings = apply_morpho(
        probable_buildings, "binary_erosion", params["building_erosion"]
    )

    false_positive = np.logical_and(
        apply_morpho(wsf[0] == 255, "binary_dilation", 10) == 0,
        urbanmask[0] > params["building_threshold"],
    )

    markers[0][probable_buildings] = params["value_classif_buildings"]
    markers[0][false_positive] = params[
        "value_classif_false_positive_buildings"
    ]

    eroded_low_veg = apply_morpho(
        vegmask[0] == 21, "binary_erosion", params["building_erosion"]
    )
    markers[0][eroded_low_veg] = params["value_classif_low_veg"]
    # careful : vegetation mask has two values for high veg !
    eroded_high_veg = apply_morpho(
        np.logical_or(vegmask[0] == 23, vegmask[0] == 22),
        "binary_erosion",
        params["building_erosion"],
    )
    markers[0][eroded_high_veg] = params["value_classif_high_veg"]

    eroded_shadow = apply_morpho(
        shadowmask[0] == 2, "binary_erosion", params["building_erosion"]
    )
    markers[0][eroded_shadow] = params["value_classif_background"]

    markers[watermask == 1] = params["value_classif_background"]

    seg = segmentation.watershed(edges, markers[0].astype(np.uint8))
    
    return seg, markers


def watershed_categorized_water(wbm, watermask, params):
    """
    Clean and apply watershed regulation for the watermask

    :param np.ndarray wbm: WBM file from post_process function
    :param np.ndarray watermask: Watermask created by the dedicated script
    :param dict params: dictionary of arguments
    :return: categorized mask
    """
    
    SEA = 1
    LAKE = 2
    RIVER = 3
    
    # 1st step : obtain a clean watermask
    nb_iter = 10
    for _ in range(nb_iter):
        clean_watermask = apply_morpho(watermask[0] == 1, "binary_opening", params["binary_opening"])
    # remove small objects in order to reduce the segmentation
    watermask_remove = apply_morpho(clean_watermask == 1, "remove_small_objects", params["minimal_size_water_area"])
    
    # 2nd step: segmentation
    # label image regions
    label_image = label(watermask_remove)
    # each instance is defined by a region
    regions = regionprops(label_image)

    # create an empty binary mask for each category of water 
    sea_mask = np.zeros(label_image.shape)
    lake_mask = np.zeros(label_image.shape)
    river_mask = np.zeros(label_image.shape)
    
    # instantiate each binary mask
    for region in regions:
        # take the center of each region
        coords = np.round(region.centroid).astype(int)
        # and apply the label of the WBM 
        if (wbm[0, coords[0], coords[1]] == SEA):
            sea_mask = np.where(label_image == region.label, sea_mask + 1, sea_mask)
        elif (wbm[0, coords[0], coords[1]] == LAKE):
            lake_mask = np.where(label_image == region.label, lake_mask + 1, lake_mask)
        # river and regions not in the WBM are considered as river
        else:
            river_mask = np.where(label_image == region.label, river_mask +1 , river_mask)          
    
    # combined the binary masks into a stack mask
    categorized_watermask = np.where(sea_mask, sea_mask*params["value_classif_sea"], 
                          np.where(lake_mask, lake_mask*params["value_classif_lake"], 
                                   np.where(river_mask, river_mask*params["value_classif_river"], watermask_remove)))  

    return categorized_watermask


def post_process(
    input_buffer: list, input_profiles: list, params: dict
) -> np.ndarray:
    """
    key_image, key_validstack, key_watermask, key_vegmask, key_urbanmask, key_shadowmask, key_wsf [, key_wbm ]
    0          1              2              3             4              5               6       [ 7 ]
    """
    input_image = input_buffer[0]
    valid_stack = input_buffer[1]
    watermask = input_buffer[2]
    vegmask = input_buffer[3]
    urbanmask = input_buffer[4]
    shadowmask = input_buffer[5]
    wsf = input_buffer[6]
    

    # 1st channel is the class, 2nd is an estimation of height class, 3rd the markers layer, for debug purpose
    stack = np.zeros((3, input_image.shape[1], input_image.shape[2]))

    # Improve buildings detection using a watershed / markers regularization
    seg, markers = watershed_regul_buildings(
        input_image, urbanmask, wsf, vegmask, watermask, shadowmask, params
    )

    clean_bare_ground = (
        morpho_clean(seg == params["value_classif_bare_ground"], params) == 1
    )
    stack[0][clean_bare_ground] = params["value_classif_bare_ground"]

    clean_buildings = (
        morpho_clean(seg == params["value_classif_buildings"], params) == 1
    )
    stack[0][clean_buildings] = params["value_classif_buildings"]

    # Note : Watermask and vegetation mask should be quite clean and don't need morpho postprocess
    stack[0][watermask[0] == 1] = params["value_classif_water"]

    low_veg = seg == params["value_classif_low_veg"]
    clean_low_veg = morpho_clean(low_veg, params) == 1
    stack[0][clean_low_veg] = params["value_classif_low_veg"]

    high_veg = seg == params["value_classif_high_veg"]
    clean_high_veg = morpho_clean(high_veg, params) == 1
    stack[0][clean_high_veg] = params["value_classif_high_veg"]

    # Apply NODATA
    stack[0][np.logical_not(valid_stack[0])] = NODATA_INT8

    # Estimation of heigth
    # Supposed to be low
    stack[1][clean_bare_ground] = LOW
    stack[1][low_veg] = LOW

    # Supposed to be high
    stack[1][clean_buildings] = HIGH
    stack[1][high_veg] = HIGH

    # No confidence in heigh
    stack[1][watermask[0] == 1] = 0
    stack[1][shadowmask[0] == 2] = 0

    stack[1][np.logical_not(valid_stack[0])] = NODATA_INT8

    # Layer 2: watermask categorized
    if params["categorized_watermask"]:
        wbm = input_buffer[7]
        categorized_watermask = watershed_categorized_water(wbm, watermask, params)
        stack[0] = np.where(categorized_watermask !=0, categorized_watermask, stack[0])

    # Markers
    stack[2] = markers
    stack[2][np.logical_not(valid_stack[0])] = NODATA_INT8

    return stack


def getarguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()

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
    group1.add_argument("-vegetationmask", help="Vegetation mask")
    group1.add_argument("-watermask", help="Water mask")
    group1.add_argument("-urbanmask", help="Urban mask probabilities")
    group1.add_argument("-shadowmask", help="Shadow mask")
    group1.add_argument("-wsf", dest="extracted_wsf", help="Extracted World Settlement Footprint raster  filename")
    group1.add_argument("-wbm", dest="extracted_wbm", help="Extracted Water Body Mask raster filename")

    group2 = parser.add_argument_group(description="*** WATERSHED OPTIONS ***")
    group2.add_argument(
        "-building_threshold",
        type=int,
        help="Threshold to consider building as detected",
    )
    group2.add_argument(
        "-building_erosion",
        type=int,
        help="Supposed buildings will be eroded by this size in the marker step",
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

    group2.add_argument("-categorized_watermask",
                        type=bool,
                        help="If true, compute a categorized watermask based on the WBM file"
                        )

    group2.add_argument(
        "-minimal_size_water_area",
        type=int,
        default=10000,
        help="Minimal area (in pixels) of water bodies"
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

    with open("args_list.json", 'w') as f:
        json.dump(arglist, f)

    return vars(args)


def slurp_stackmask(main_config: str, logs_to_file: bool, debug: bool, user_config: str, file_vhr: str,
                    valid_stack: bool, vegetationmask: str, watermask: str, urbanmask: str, shadowmask: str,
                    extracted_wsf: str, extracted_wbm: str, building_threshold: int, building_erosion: int, bonus_gt: int,
                    malus_shadow: int, stackmask: str, value_classif_low_veg: int, value_classif_high_veg: int,
                    value_classif_water: int, value_classif_buildings: int, value_classif_bare_ground: int,
                    value_classif_false_positive_buildings: int, value_classif_background: int, n_workers: int, tile_max_size: int,
                    multiproc_context: str, categorized_watermask: bool, minimal_size_water_area: int):
    """
    Main API to compute urban mask.
    """
    # Read the JSON files
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
        # If the parameter from the CLI is not None, we update argsdict with the value from the CLI
        if locals()[param] is not None:
            argsdict[param] = locals()[param]

    logger.info("--" * 50)
    logger.info("SLURP - Stack masks\n")
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
            key_image = eoscale_manager.open_raster(raster_path=args.file_vhr)

            key_watermask = eoscale_manager.open_raster(
                raster_path=args.watermask
            )
            key_vegmask = eoscale_manager.open_raster(
                raster_path=args.vegetationmask
            )
            key_urbanmask = eoscale_manager.open_raster(
                raster_path=args.urbanmask
            )
            key_shadowmask = eoscale_manager.open_raster(
                raster_path=args.shadowmask
            )
            key_wsf = eoscale_manager.open_raster(
                raster_path=args.extracted_wsf
            )
            key_validstack = eoscale_manager.open_raster(
                raster_path=args.valid_stack
            )
            if args.categorized_watermask:
                key_wbm = eoscale_manager.open_raster(
                    raster_path=args.extracted_wbm
                )
                inputs_final = [key_image,
                                key_validstack,
                                key_watermask,
                                key_vegmask,
                                key_urbanmask,
                                key_shadowmask,
                                key_wsf,
                                key_wbm]
            else:
                inputs_final = [key_image,
                                key_validstack,
                                key_watermask,
                                key_vegmask,
                                key_urbanmask,
                                key_shadowmask,
                                key_wsf]
            args.nodata_vhr = 0  # TODO : get nodata value from image profile

            final_mask = eoexe.n_images_to_m_images_filter(
                inputs=inputs_final,
                image_filter=post_process,
                filter_parameters=vars(args),
                generate_output_profiles=eo_utils.three_uint8_profile,
                stable_margin=200,
                context_manager=eoscale_manager,
                multiproc_context=args.multiproc_context,
                filter_desc="Post processing...",
            )

            eoscale_manager.write(key=final_mask[0], img_path=args.stackmask)

            t1 = time.time()
            logger.info(
                f"**** Stack masks for {args.file_vhr} (saved as {args.stackmask}) ****"
            )
            logger.info("Total time (user)       :\t" + utils.convert_time(t1 - t0))

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
    Main function to run the stack mask computation.
    It parses the command line arguments and calls the slurp_stackmask function.
    """
    args = getarguments()
    slurp_stackmask(**args)

if __name__ == "__main__":
    main()
