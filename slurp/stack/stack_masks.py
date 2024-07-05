#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script stacks existing masks
"""

import argparse
import traceback
import numpy as np
import time

from skimage.morphology import (binary_closing, binary_opening, binary_dilation, binary_erosion,
                                remove_small_holes, remove_small_objects, disk)
from skimage.filters import sobel
from skimage import segmentation

from slurp.tools import eoscale_utils as eo_utils
from slurp.tools import io_utils
import eoscale.manager as eom
import eoscale.eo_executors as eoexe


"""
Final mask values 
- 1st layer : class
- 2nd layer : estimation of elevation
"""
BACKGROUND = 11

LOW_VEG = 1
HIGH_VEG = 2
WATER = 3
BUILDINGS = 4
UNDEF_WATER_URBAN = 5  # Prediction of water, but also urban
BARE_GROUND = 6
UNDEF_URBAN_BARE_GROUND = (
    7  # Prediction of urban but smooth area (could be bare ground)
)
WATER_PRED = 8  # Prediction of water, but not in Peckel database
SHADOW = 9
BUILDINGS_FALSE_POSITIVE = 10


# Elevation estimation in 2nd layer
LOW = 1
HIGH = 2
NOT_CONFIDENT = 0

NODATA = 255


def watershed_regul_buildings(input_image, urban_proba, wsf, vegmask, watermask, shadowmask, params):
    # Compute mono image from RGB image
    im_mono = 0.29*input_image[0] + 0.58*input_image[1] + 0.114*input_image[2]
    edges = sobel(im_mono)

    markers = np.zeros((1, input_image.shape[1], input_image.shape[2]))
    
    # We set markers by reverse order of confidence
    eroded_bare_ground = binary_erosion(vegmask[0] == 11, disk(params["building_erosion"]))
    markers[0][eroded_bare_ground] = BARE_GROUND
    
    ground_truth_eroded = binary_erosion(wsf[0] == 255, disk(params["building_erosion"]))

    # Bonus for pixels above ground truth
    urban_proba[0][ground_truth_eroded] += params["bonus_gt"]
    # Malus for pixels in shadow areas
    urban_proba[0][shadowmask[0] == 2] -= params["malus_shadow"]
    probable_buildings = np.logical_and(ground_truth_eroded, urban_proba[0] > params["building_threshold"])
    probable_buildings = binary_erosion(probable_buildings, disk(params["building_erosion"]))
    
    false_positive = np.logical_and(
        binary_dilation(wsf[0] == 255, disk(10)) == 0,
        urban_proba[0] > params["building_threshold"]
    )
    
    markers[0][probable_buildings] = BUILDINGS
    markers[0][false_positive] = BUILDINGS_FALSE_POSITIVE

    markers[0][binary_erosion(vegmask[0] == 21, disk(params["building_erosion"]))] = LOW_VEG
    markers[0][binary_erosion(vegmask[0] == 22, disk(params["building_erosion"]))] = HIGH_VEG
    markers[0][binary_erosion(vegmask[0] == 23, disk(params["building_erosion"]))] = HIGH_VEG
    
    # markers[shadowmask == 2] = BACKGROUND
    markers[0][binary_erosion(shadowmask[0] == 2, disk(params["building_erosion"]))] = BACKGROUND
    
    markers[watermask == 1] = BACKGROUND

    seg = segmentation.watershed(edges, markers[0].astype(np.uint8))
    
    return seg, markers


def morpho_clean(im_classif, params):
        
    if params["binary_closing"]:
        # Closing can remove small dark spots (i.e. “pepper”) and connect small bright cracks.
        im_classif = binary_closing(im_classif, disk(params["binary_closing"])).astype(np.uint8)

    if params["binary_opening"]:
        # Opening can remove small bright spots (i.e. “salt”) and connect small dark cracks.
        im_classif = binary_opening(im_classif, disk(params["binary_opening"])).astype(np.uint8)

    if params["remove_small_holes"]:
        im_classif = remove_small_holes(im_classif.astype(bool), params["remove_small_holes"], connectivity=2)
        
    if params["remove_small_objects"]:
        im_classif = remove_small_objects(im_classif, params["remove_small_objects"], connectivity=2)
        
    return im_classif.astype(np.uint8)
    

def post_process(inputBuffer: list,  input_profiles: list,  params: dict) -> np.ndarray:
    """
    key_image, key_validstack, key_watermask, key_waterpred, key_vegmask, key_urban_proba, key_shadowmask, key_wsf
    0          1              2              3              4            5                6               7   
    """
    input_image = inputBuffer[0]
    valid_stack = inputBuffer[1]
    watermask   = inputBuffer[2]
    vegmask     = inputBuffer[4]
    urban_proba = inputBuffer[5]
    shadowmask  = inputBuffer[6]
    wsf = inputBuffer[7]

    # 1st channel is the class, 2nd is an estimation of height class, 3rd the markers layer, for debug purpose
    stack = np.zeros((3, input_image.shape[1], input_image.shape[2]))

    # Improve buildings detection using a watershed / markers regularization
    segmentation, markers = watershed_regul_buildings(
        input_image, urban_proba, wsf, vegmask, watermask, shadowmask, params
    )

    clean_bare_ground = morpho_clean(segmentation == BARE_GROUND, params) == 1
    stack[0][clean_bare_ground] = BARE_GROUND

    clean_buildings = morpho_clean(segmentation == BUILDINGS, params) == 1
    stack[0][clean_buildings] = BUILDINGS

    # Note : Watermask and vegetation mask should be quite clean and don't need morpho postprocess
    stack[0][watermask[0] == 1] = WATER

    low_veg = segmentation == LOW_VEG
    clean_low_veg = morpho_clean(low_veg, params) == 1
    stack[0][clean_low_veg] = LOW_VEG

    high_veg = segmentation == HIGH_VEG
    clean_high_veg = morpho_clean(high_veg, params) == 1
    stack[0][clean_high_veg] = HIGH_VEG

    # Apply NODATA
    stack[0][np.logical_not(valid_stack[0])] = NODATA

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
    
    stack[1][np.logical_not(valid_stack[0])] = NODATA

    # Markers
    stack[2] = markers
    stack[2][np.logical_not(valid_stack[0])] = NODATA

    return stack


def getarguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("main_config", help="First JSON file, load basis arguments")
    parser.add_argument("-user_config", help="Second JSON file, overload basis arguments if keys are the same")
    parser.add_argument("-file_vhr", help="VHR input image")
    parser.add_argument("-stackmask", help="Final mask")
    parser.add_argument("-vegetationmask", help="Vegetation mask")
    parser.add_argument("-vegmask_max_value", type=int, action="store",
                        help="Vegetation mask value for vegetated areas : all pixels with lower value will be predicted")
    parser.add_argument("-watermask", help="Water mask (filtered mask)")
    parser.add_argument("-waterpred", help="Water mask (prediction)")
    parser.add_argument("-urban_proba", help="Urban mask probabilities")
    parser.add_argument("-building_threshold", type=int, action="store",
                        help="Threshold to consider building as detected (70 by default)")
    parser.add_argument("-shadowmask", help="Shadow mask")
    parser.add_argument("-valid", action="store", dest="valid_stack", help="Validity mask")
     
    parser.add_argument("-extracted_wsf", help="World Settlement Footprint raster")
    parser.add_argument("-mnh", help="Height elevation model")
    parser.add_argument("-binary_closing", type=int, action="store",
                        help="Size of square structuring element (clean BUILDING / BARE_GROUND classes)")
    parser.add_argument("-binary_opening", type=int, action="store",
                        help="Size of square structuring element (clean BUILDING / BARE_GROUND classes)")
    parser.add_argument("-building_erosion", type=int, action="store",
                        help="Supposed buildings will be eroded by this size in the marker step")

    parser.add_argument("-bonus_gt", type=int, action="store",
                        help="Bonus for pixels covered by GT, in the watershed regularization step "
                             "(ex : +30 to improve discrimination between building and background)")
    parser.add_argument("-malus_shadow", type=int, action="store",
                        help="Malus for pixels in shadow, in the watershed regularization step")
     
    parser.add_argument("-remove_small_objects", type=int, action="store",
                        help="The minimum area, in pixels, of the objects to detect")
    parser.add_argument("-remove_small_holes", type=int, action="store",
                        help="The minimum area, in pixels, of the holes to fill")
    
    parser.add_argument("-n_workers", type=int, action="store", help="Nb of CPU")

    return parser.parse_args()


def main():
    argparse_dict = vars(getarguments())

    # Read the JSON files
    keys = ['input', 'aux_layers', 'masks', 'ressources', 'stack']
    argsdict = io_utils.read_json(argparse_dict["main_config"], keys, argparse_dict.get("user_config"))

    # Overload with manually passed arguments if not None
    for key in argparse_dict.keys():
        if argparse_dict[key] is not None:
            argsdict[key] = argparse_dict[key]

    print("JSON data loaded:")
    print(argsdict)
    args = argparse.Namespace(**argsdict)
    
    with eom.EOContextManager(nb_workers=args.n_workers, tile_mode=True) as eoscale_manager:
        try:
            t0 = time.time()
            key_image = eoscale_manager.open_raster(raster_path=args.file_vhr)
            key_watermask = eoscale_manager.open_raster(raster_path=args.watermask)
            key_waterpred = key_watermask  # eoscale_manager.open_raster(raster_path=args.waterpred)
            key_vegmask = eoscale_manager.open_raster(raster_path=args.vegetationmask)
            key_urban_proba = eoscale_manager.open_raster(raster_path=args.urban_proba)
            key_shadowmask = eoscale_manager.open_raster(raster_path=args.shadowmask)
            key_wsf = eoscale_manager.open_raster(raster_path=args.extracted_wsf)
            key_validstack = eoscale_manager.open_raster(raster_path=args.valid_stack)
                
            args.nodata_vhr = 0  # TODO : get nodata value from image profile

            inputs_final = [key_image, key_validstack, key_watermask, key_waterpred, key_vegmask, key_urban_proba, key_shadowmask, key_wsf]
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
