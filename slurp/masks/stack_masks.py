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
import numpy as np
import traceback
import time

from os import path, makedirs
from skimage.filters import sobel
from skimage import segmentation
from skimage.util import map_array

import eoscale.manager as eom
import eoscale.eo_executors as eoexe
from slurp.post_process.morphology import morpho_clean, apply_morpho
from slurp.tools import eoscale_utils as eo_utils
from slurp.tools import io_utils
from slurp.tools.constant import NODATA_int8, LOW, HIGH


def watershed_regul_buildings(input_image, urbanmask, wsf, vegmask, watermask, shadowmask, params):
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
    #im_mono = 0.29*input_image[0] + 0.58*input_image[1] + 0.114*input_image[2]
    im_mono = 0.3*input_image[0] + 0.3*input_image[1] + 0.3*input_image[3]
    edges = sobel(im_mono)

    markers = np.zeros((1, input_image.shape[1], input_image.shape[2]))
    
    # We set markers by reverse order of confidence
    eroded_bare_ground = apply_morpho(vegmask[0] == 11, "binary_erosion", params["building_erosion"])
    markers[0][eroded_bare_ground] = params["value_classif_bare_ground"]
    
    ground_truth_eroded = apply_morpho(wsf[0] == 255, "binary_erosion", params["building_erosion"])

    # Bonus for pixels above ground truth
    urbanmask[0][ground_truth_eroded] += params["bonus_gt"]
    # Malus for pixels in shadow areas
    urbanmask[0][shadowmask[0] == 2] -= params["malus_shadow"]
    probable_buildings = np.logical_and(ground_truth_eroded, urbanmask[0] > params["building_threshold"])
    probable_buildings = apply_morpho(probable_buildings, "binary_erosion", params["building_erosion"])
    
    false_positive = np.logical_and(
        apply_morpho(wsf[0] == 255, "binary_dilation", 10) == 0,
        urbanmask[0] > params["building_threshold"]
    )
    
    markers[0][probable_buildings] = params["value_classif_buildings"]
    markers[0][false_positive] = params["value_classif_false_positive_buildings"]

    eroded_low_veg = apply_morpho(vegmask[0] == 21, "binary_erosion", params["building_erosion"])
    markers[0][eroded_low_veg] = params["value_classif_low_veg"]
    # careful : vegetation mask has two values for high veg !
    eroded_high_veg = apply_morpho(np.logical_or(vegmask[0] == 23, vegmask[0] == 22), "binary_erosion", params["building_erosion"])
    markers[0][eroded_high_veg] = params["value_classif_high_veg"]
    
    eroded_shadow = apply_morpho(shadowmask[0] == 2, "binary_erosion", params["building_erosion"])
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

    not_water_value = 0
    sea_value_wbm = 1
    lake_value_wbm = 2
    river_value_wbm = 3
    water_value_watermask = 1
    water_unknown_value = 4

    # Intersection SLURP watermask and WBM mask
    # No water in WBM, water in watermask
    no_water_wbm_water_watermask = np.logical_and(watermask[0] == water_value_watermask, wbm[0] == not_water_value)
    # Water in WBM, no water in watermask
    water_wbm_no_water_watermask = np.logical_and(watermask[0] == not_water_value, wbm[0] != not_water_value)
    # Compute the intersection
    water_not_defined_watermask = np.where(no_water_wbm_water_watermask, water_unknown_value, wbm[0])
    intersection_wbm_watermask = np.where(water_wbm_no_water_watermask, watermask[0], water_not_defined_watermask)

    # Remove holes and small objects from SLURP watermask
    watermask_without_frame = map_array(
        input_arr=watermask[0],
        input_vals=np.array([0, 1, 255]),
        output_vals=np.array([0, 1, 0]))
    watermask_remove_small_holes_filled = apply_morpho(watermask_without_frame.astype(bool),"remove_small_holes",
                                                       params["minimal_size_water_area"])
    watermask_cleaned = apply_morpho(watermask_remove_small_holes_filled.astype(bool), "remove_small_objects",
                                                       params["minimal_size_water_area"])


    # Remove holes and small objects from intersection SLURP and WBM
    binary_objects = intersection_wbm_watermask.astype(bool)
    binary_filled = apply_morpho(binary_objects,"remove_small_holes", params["minimal_size_water_area"])
    binary_filled = apply_morpho(binary_filled, "remove_small_objects", params["minimal_size_water_area"])
    # Get the intersection classified
    objects_filled = segmentation.watershed(
        binary_filled, intersection_wbm_watermask.astype(int), mask=binary_filled
    )

    # Compute markers
    markers = np.zeros((1, intersection_wbm_watermask.shape[0], intersection_wbm_watermask.shape[1]))
    not_water = apply_morpho(objects_filled == not_water_value, "binary_erosion", 5)
    sea = apply_morpho(objects_filled == sea_value_wbm, "binary_dilation", 10)
    lake = apply_morpho(objects_filled == lake_value_wbm, "binary_dilation", 10)
    river = apply_morpho(objects_filled == river_value_wbm, "binary_dilation", 10)
    water_unknown_full = apply_morpho(objects_filled == water_unknown_value,"remove_small_objects",
                                      params["minimal_size_water_area"])
    water_unknown = apply_morpho(water_unknown_full,"binary_erosion", 20)
    markers[0][not_water] = not_water_value
    markers[0][water_unknown] = params["value_classif_water"]
    markers[0][sea] = params["value_classif_sea"]
    markers[0][lake] = params["value_classif_lake"]
    markers[0][river] = params["value_classif_river"]

    # Segmentation
    seg = segmentation.watershed(watermask_cleaned, markers=markers[0].astype(np.uint8),
                                 mask=watermask_cleaned)

    mask = (watermask[0] != 255)
    categorized_watermask = np.where(mask, seg, NODATA_int8)

    return categorized_watermask

def post_process(input_buffer: list,  input_profiles: list,  params: dict) -> np.ndarray:
    """
    key_image, key_validstack, key_watermask, key_vegmask, key_urbanmask, key_shadowmask, key_wsf
    0          1              2              3             4              5               6
    """
    input_image = input_buffer[0]
    valid_stack = input_buffer[1]
    watermask   = input_buffer[2]
    vegmask     = input_buffer[3]
    urbanmask   = input_buffer[4]
    shadowmask  = input_buffer[5]
    wsf = input_buffer[6]


    # 1st channel is the class, 2nd is an estimation of height class, 3rd the markers layer, for debug purpose
    stack = np.zeros((3, input_image.shape[1], input_image.shape[2]))

    # Improve buildings detection using a watershed / markers regularization
    seg, markers = watershed_regul_buildings(
        input_image, urbanmask, wsf, vegmask, watermask, shadowmask, params
    )

    clean_bare_ground = morpho_clean(seg == params["value_classif_bare_ground"], params) == 1
    stack[0][clean_bare_ground] = params["value_classif_bare_ground"]

    clean_buildings = morpho_clean(seg == params["value_classif_buildings"], params) == 1
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
    stack[0][np.logical_not(valid_stack[0])] = NODATA_int8

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
    
    stack[1][np.logical_not(valid_stack[0])] = NODATA_int8

    # Layer 2: watermask categorized
    if params["extracted_wbm"]=="out/wbm.tif":
        wbm = input_buffer[7]
        stack[2] = watershed_categorized_water(wbm, watermask, params)

    """
    # Layer 3 : segmentation from watershed, before morpho/clean
    stack[3] = seg
    stack[3][np.logical_not(valid_stack[0])] = NODATA_int8

    # Layer 4 : compute simple urban mask with proba > threshold + morpho clean phase
    
    buildings = np.where(urbanmask > params["building_threshold"],1,0)
    stack[4] = morpho_clean(buildings[0], params)
    stack[4][np.logical_not(valid_stack[0])] = NODATA_int8
    """

    return stack


def getarguments():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser()

    parser.add_argument("main_config", help="First JSON file, load basis arguments")

    group1 = parser.add_argument_group(description="*** INPUT FILES ***")
    group1.add_argument("-user_config", help="Second JSON file, overload basis arguments if keys are the same")
    group1.add_argument("-file_vhr", help="Input 4 bands VHR image")
    group1.add_argument("-valid", dest="valid_stack", help="Validity mask")
    group1.add_argument("-vegetationmask", help="Vegetation mask")
    group1.add_argument("-watermask", help="Water mask")
    group1.add_argument("-urbanmask", help="Urban mask probabilities")
    group1.add_argument("-shadowmask", help="Shadow mask")
    group1.add_argument("-wsf", dest="extracted_wsf", help="Extracted World Settlement Footprint raster  filename")
    group1.add_argument("-wbm", dest="extracted_wbm", help="Extracted Water Body Mask raster filename")

    group2 = parser.add_argument_group(description="*** WATERSHED OPTIONS ***")
    group2.add_argument("-building_threshold", type=int, help="Threshold to consider building as detected")
    group2.add_argument("-building_erosion", type=int,
                        help="Supposed buildings will be eroded by this size in the marker step")
    group2.add_argument("-bonus_gt", type=int,
                        help="Bonus for pixels covered by GT, in the watershed regularization step "
                             "(ex : +30 to improve discrimination between building and background)")
    group2.add_argument("-malus_shadow", type=int,
                        help="Value of the malus for pixels in shadow, in the watershed regularization step")

    group4 = parser.add_argument_group(description="*** OUTPUT FILE ***")
    group4.add_argument("-stackmask", help="Output Final mask filename")
    group4.add_argument("-low_veg", dest="value_classif_low_veg", type=int,
                        help="Output classification value for low vegetation")
    group4.add_argument("-high_veg", dest="value_classif_high_veg", type=int,
                        help="Output classification value for high vegetation")
    group4.add_argument("-water", dest="value_classif_water", type=int, help="Output classification value for water")
    group4.add_argument("-buildings", dest="value_classif_buildings", type=int,
                        help="Output classification value for buildings")
    group4.add_argument("-bare_ground", dest="value_classif_bare_ground", type=int,
                        help="Output classification value for bare ground")
    group4.add_argument("-false_pos_buildings", dest="value_classif_false_positive_buildings", type=int,
                        help="Output classification value for buildings false positive")
    group4.add_argument("-background", dest="value_classif_background", type=int,
                        help="Output classification value for background")

    group5 = parser.add_argument_group(description="*** PARALLEL COMPUTING ***")
    group5.add_argument("-n_workers", type=int, help="Number of CPU")

    return parser.parse_args()


def main():
    """Main function that stacks the masks compute before"""
    
    argparse_dict = vars(getarguments())

    # Read the JSON files
    keys = ["input", "aux_layers", "masks", "resources", "post_process", "stack"]
    argsdict = io_utils.read_json(argparse_dict["main_config"], keys, argparse_dict.get("user_config"))

    # Overload with manually passed arguments if not None
    for key in argparse_dict.keys():
        if argparse_dict[key] is not None:
            argsdict[key] = argparse_dict[key]

    print("JSON data loaded:")
    print(argsdict)
    args = argparse.Namespace(**argsdict)
    
    # Create output folder
    makedirs(path.dirname(args.stackmask), exist_ok=True)
    
    # Mask calculation     
    with eom.EOContextManager(nb_workers=args.n_workers, tile_mode=True) as eoscale_manager:
        try:
            t0 = time.time()
            key_image = eoscale_manager.open_raster(raster_path=args.file_vhr)
            key_watermask = eoscale_manager.open_raster(raster_path=args.watermask)
            key_vegmask = eoscale_manager.open_raster(raster_path=args.vegetationmask)
            key_urbanmask = eoscale_manager.open_raster(raster_path=args.urbanmask)
            key_shadowmask = eoscale_manager.open_raster(raster_path=args.shadowmask)
            key_wsf = eoscale_manager.open_raster(raster_path=args.extracted_wsf)
            key_validstack = eoscale_manager.open_raster(raster_path=args.valid_stack)
            if args.extracted_wbm is not None:
                key_wbm = eoscale_manager.open_raster(raster_path=args.extracted_wbm)
                inputs_final = [key_image, key_validstack, key_watermask, key_vegmask, key_urbanmask, key_shadowmask,
                                key_wsf, key_wbm]
            else:
                inputs_final = [key_image, key_validstack, key_watermask, key_vegmask, key_urbanmask, key_shadowmask,
                                key_wsf]

            args.nodata_vhr = 0  # TODO : get nodata value from image profile

            final_mask = eoexe.n_images_to_m_images_filter(inputs=inputs_final,
                                                           image_filter=post_process,
                                                           filter_parameters=vars(args),
                                                           generate_output_profiles=eo_utils.three_uint8_profile,
                                                           stable_margin=200,
                                                           context_manager=eoscale_manager,
                                                           multiproc_context="fork",
                                                           filter_desc="Post processing...")
                
            eoscale_manager.write(key=final_mask[0], img_path=args.stackmask)
                
            t1 = time.time()

            print("Total time (user)       :\t" + str(t1-t0))
                
        except FileNotFoundError as fnfe_exception:
            print("FileNotFoundError", fnfe_exception)

        except PermissionError as pe_exception:
            print("PermissionError", pe_exception)

        except ArithmeticError as ae_exception:
            print("ArithmeticError", ae_exception)

        except MemoryError as me_exception:
            print("MemoryError", me_exception)

        except Exception as exception:  # pylint: disable=broad-except
            print("oups...", exception)
            traceback.print_exc()

                
if __name__ == "__main__":
    main()
