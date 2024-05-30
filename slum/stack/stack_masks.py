#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script stacks existing masks
"""

import argparse
import rasterio as rio
import numpy as np
import argparse
import os
from skimage.morphology import (
    area_closing,
    binary_closing,
    binary_opening,
    binary_dilation,
    binary_erosion,
    diameter_closing,
    remove_small_holes,
    remove_small_objects,
    square, disk
)
from skimage.filters import sobel
from skimage import segmentation

import eoscale.manager as eom
import eoscale.eo_executors as eoexe
import time

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

def compute_valid_stack(inputBuffer: list, 
            input_profiles: list, 
            args: dict) -> list:
    #inputBuffer =  [key_phr, mask_nocloud_key] 
    # Valid_phr (boolean numpy array, True = valid data, False = no data)
    valid_phr = np.logical_and.reduce(inputBuffer[0] != args.nodata_vhr, axis=0)
    valid_stack = np.logical_and(valid_phr, inputBuffer[1])
    
    return valid_stack

def post_process_profile(input_profiles: list, map_params):
    profile = input_profiles[0]
    profile['count'] = 3
    profile['dtype'] =np.uint8
    profile["compress"] = "lzw"
    profile["nodata"] = 255

    return profile

def single_uint8_profile(input_profiles: list, map_params):
    profile = input_profiles[0]
    profile["count"]= 1
    profile["dtype"]= np.uint8
    profile["compress"] = "lzw"
    profile["nodata"] = 255
    
    return profile
def single_bool_profile(input_profiles: list, map_params):
    profile = input_profiles[0]
    profile['count']=1
    profile['dtype']=bool
    profile["compress"] = "lzw"

    return profile

def watershed_regul_buildings(input_image, urban_proba, wsf, vegmask, watermask, shadowmask, params):
    # Compute mono image from RGB image
    im_mono = 0.29*input_image[0] + 0.58*input_image[1] + 0.114*input_image[2]
    edges = sobel(im_mono)

    markers = np.zeros((1,input_image.shape[1],input_image.shape[2]))
    
    # We set markers by reverse order of confidence
    eroded_bare_ground = binary_erosion(vegmask[0] == 11, disk(params.building_erosion))
    markers[0][eroded_bare_ground] = BARE_GROUND
    
    ground_truth_eroded = binary_erosion(wsf[0]==255, disk(params.building_erosion)) 

    # Bonus for pixels above ground truth
    urban_proba[0][ground_truth_eroded] += 30
    # Malus for pixels in shadow areas
    urban_proba[0][shadowmask[0]==2] -= 30
    probable_buildings = np.logical_and(ground_truth_eroded, urban_proba[0] > params.building_threshold)
    probable_buildings = binary_erosion(probable_buildings, disk(params.building_erosion))
    
    false_positive = np.logical_and(binary_dilation(wsf[0]==255, disk(10)) == 0, urban_proba[0] > params.building_threshold)
    
    markers[0][probable_buildings] = BUILDINGS
    markers[0][false_positive] = BUILDINGS_FALSE_POSITIVE

    """
    markers[vegmask == 21] = LOW_VEG
    markers[vegmask == 22] = HIGH_VEG
    markers[vegmask == 23] = HIGH_VEG
    """
    
    markers[0][binary_erosion(vegmask[0] == 21,disk(params.building_erosion))] = LOW_VEG
    markers[0][binary_erosion(vegmask[0] == 22,disk(params.building_erosion))] = HIGH_VEG
    markers[0][binary_erosion(vegmask[0] == 23,disk(params.building_erosion))] = HIGH_VEG
    
    #markers[shadowmask == 2] = BACKGROUND
    markers[0][binary_erosion(shadowmask[0] == 2,disk(params.building_erosion))] = BACKGROUND
    
    markers[watermask == 1] = BACKGROUND

    #print(f"DBG {markers[0]}")
    seg = segmentation.watershed(edges, markers[0].astype(np.uint8))
    
    return seg, markers

def morpho_clean(im_classif, args):
    
        
    if args.binary_closing:
        # Closing can remove small dark spots (i.e. “pepper”) and connect small bright cracks.
        im_classif = binary_closing(im_classif, disk(args.binary_closing)).astype(np.uint8)

    if args.binary_opening:
        # Opening can remove small bright spots (i.e. “salt”) and connect small dark cracks.
        im_classif = binary_opening(im_classif, disk(args.binary_opening)).astype(np.uint8)

    if args.remove_small_holes:
        im_classif = remove_small_holes(
            im_classif.astype(bool), args.remove_small_holes, connectivity=2
        )
        
    if args.remove_small_objects:
        im_classif = remove_small_objects(
            im_classif, args.remove_small_objects, connectivity=2
        )

        
    return im_classif.astype(np.uint8)
    
    

def post_process(inputBuffer: list, 
                input_profiles: list, 
                params: dict) -> list:
    """
    key_image, key_validstak, key_watermask, key_waterpred, key_vegmask, key_urban_proba, key_shadowmask, key_wsf
    0          1              2              3              4            5                6               7   
    """
    input_image = inputBuffer[0]
    valid_stack = inputBuffer[1]
    watermask   = inputBuffer[2]
    vegmask     = inputBuffer[4]
    urban_proba   = inputBuffer[5]
    shadowmask  = inputBuffer[6]
    wsf = inputBuffer[7]

    # 1st channel is the class, 2nd is an estimation of height class, 3rd the markers layer, for debug purpose
    stack = np.zeros((3,input_image.shape[1],input_image.shape[2]))

    # Improve buildings detection using a watershed / markers regularization
    segmented_buildings, markers = watershed_regul_buildings(input_image, urban_proba, wsf, vegmask, watermask, shadowmask, params)

    clean_bare_ground = morpho_clean(vegmask[0] == 11, params) == 1
    stack[0][clean_bare_ground] = BARE_GROUND

    clean_buildings = morpho_clean(segmented_buildings==BUILDINGS, params)==1
    stack[0][clean_buildings] = BUILDINGS

    # Note : Watermask and vegetation mask should be quite clean and don't need morpho postprocess
    stack[0][watermask[0] == 1] = WATER

    low_veg = vegmask[0] == 21
    stack[0][low_veg] = LOW_VEG

    high_veg = np.logical_or(vegmask[0] == 22, vegmask[0] == 23)
    stack[0][high_veg] = HIGH_VEG

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

    # Debug
    stack[1] = markers
    stack[2] = segmented_buildings
    stack[2][np.logical_not(valid_stack[0])] = NODATA
    

    return stack
    


def getarguments():
     parser = argparse.ArgumentParser()
     parser.add_argument("im", help="VHR input image")
     parser.add_argument("mask", help="Final mask")
     parser.add_argument("-vegmask", default=None, help="Vegetation mask")
     parser.add_argument(
        "-vegmask_max_value",
        required=False,
        type=int,
        default=21,
        action="store",
        dest="vegmask_max_value",
        help="Vegetation mask value for vegetated areas : all pixels with lower value will be predicted"
    )

     parser.add_argument(
         "-watermask", default=None, help="Water mask (filtered mask)"
     )
     parser.add_argument(
         "-waterpred", default=None, help="Water mask (prediction)"
     )
     parser.add_argument("-urban_proba", default=None, help="Urban mask probabilities")
     parser.add_argument(
         "-building_threshold",
         type=int,
         default=70,
         required=False,
         action="store",
         dest="building_threshold",
         help="Threshold to consider building as detected (70 by default)"
     )
     parser.add_argument("-shadowmask", default=None, help="Shadow mask")
     
     parser.add_argument("-wsf", default="", help="World Settlement Footprint raster")
     parser.add_argument("-mnh", default="", help="Height elevation model")
     parser.add_argument(
         "-binary_closing",
         type=int,
         default=0,
         required=False,
         action="store",
         dest="binary_closing",
         help="Size of square structuring element (clean BUILDING / BARE_GROUND classes)"
     )
     
     parser.add_argument(
         "-binary_opening",
         type=int,
         default=0,
         required=False,
         action="store",
         dest="binary_opening",
         help="Size of square structuring element (clean BUILDING / BARE_GROUND classes)"
     )

     parser.add_argument(
         "-building_erosion",
         type=int,
         default=0,
         required=False,
         action="store",
         dest="building_erosion",
         help="Supposed buildings will be eroded by this size in the marker step"
     )
     
     parser.add_argument(
         "-remove_small_objects",
         type=int,
         default=0,
         required=False,
         action="store",
         dest="remove_small_objects",
         help="The minimum area, in pixels, of the objects to detect",
     )
    
     parser.add_argument(
         "-remove_small_holes",
         type=int,
         default=0,
         required=False,
         action="store",
         dest="remove_small_holes",
         help="The minimum area, in pixels, of the holes to fill",
     )
    
     parser.add_argument(
        "-n_workers",
        type=int,
        default=8,
        required=False,
        action="store",
        dest="nb_workers",
        help="Nb of CPU"
     )

     parser.add_argument(
        "-cloud_gml",
        required=False,
        action="store",
        dest="file_cloud_gml",
        help="Cloud file in .GML format",
     )
     return parser.parse_args()


def main():
    try:
        args = getarguments()
        print("args")
        with eom.EOContextManager(nb_workers = args.nb_workers, tile_mode = True) as eoscale_manager:
            try:
                t0 = time.time()
                key_image = eoscale_manager.open_raster(raster_path = args.im)

                key_watermask = eoscale_manager.open_raster(raster_path = args.watermask)
                key_waterpred = eoscale_manager.open_raster(raster_path = args.waterpred)
                key_vegmask = eoscale_manager.open_raster(raster_path = args.vegmask)
                key_urban_proba = eoscale_manager.open_raster(raster_path = args.urban_proba)
                key_shadowmask = eoscale_manager.open_raster(raster_path = args.shadowmask)
                key_wsf = eoscale_manager.open_raster(raster_path = args.wsf)

                args.nodata_vhr = 0 # TODO : get nodata value from image profile
                
                # Get cloud mask if any
                if args.file_cloud_gml:
                    cloud_mask_array = np.logical_not(
                        cloud_from_gml(args.file_cloud_gml, args.im)   
                    )
                    #save cloud mask
                    save_image(cloud_mask_array,
                               join(dirname(args.mask), "nocloud.tif"),
                               args.crs,
                               args.transform,
                               None,
                               args.rpc,
                               tags=args.__dict__,
                            )
                    mask_nocloud_key = eoscale_manager.open_raster(raster_path = join(dirname(args.file_classif), "nocloud.tif"))   
                else:
                    # Get profile from im_phr
                    profile = eoscale_manager.get_profile(key_image)
                    profile["count"] = 1
                    profile["dtype"] = np.uint8
                    mask_nocloud_key = eoscale_manager.create_image(profile)
                    eoscale_manager.get_array(key=mask_nocloud_key).fill(1)

                    
                key_validstack = eoexe.n_images_to_m_images_filter(inputs = [key_image, mask_nocloud_key],
                                                                   image_filter = compute_valid_stack,   
                                                                   filter_parameters=args,
                                                                   generate_output_profiles = single_bool_profile,
                                                                   stable_margin= 0,
                                                                   context_manager = eoscale_manager,
                                                                   multiproc_context= "fork",
                                                                   filter_desc= "Valid stack processing...") 
                
                
                final_mask = eoexe.n_images_to_m_images_filter(inputs =
                                                               [key_image, key_validstack[0], key_watermask, key_waterpred, key_vegmask, key_urban_proba, key_shadowmask, key_wsf],
                                                               image_filter = post_process,
                                                               filter_parameters= args,
                                                               generate_output_profiles = post_process_profile,
                                                               stable_margin= 200,
                                                               context_manager = eoscale_manager,
                                                               multiproc_context= "fork",
                                                               filter_desc= "Post processing...")
                
                eoscale_manager.write(key = final_mask[0], img_path = args.mask)
                
                t1 = time.time()

                print("Total time (user)       :\t"+str(t1-t0))
                
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
