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

BACKGROUND = 0

# Elevation estimation in 2nd layer
LOW = 1
HIGH = 2
NOT_CONFIDENT = 0

NODATA = 255

def hello_world():
    print("Hello World")

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



def watershed_regul(args, clean_predict, inputBuffer):    
    # inputBuffer = [key_predict[0],key_phr, key_watermask, key_vegmask, key_shadowmask, gt_key, valid_stack_key[0]]
    #                   0              1          2             3              4            5         6
    
    # Compute mono image from RGB image
    im_mono = 0.29*inputBuffer[1][0] + 0.58*inputBuffer[1][1] + 0.114*inputBuffer[1][2]
       
    # compute gradient
    edges = sobel(im_mono)

    del im_mono

    # markers map : -1, 1 and 2 : probable background, buildings or false positive
    # inputBuffer[0] = proba of building class
    markers = np.zeros_like(inputBuffer[0][0])

    """
    weak_detection = np.logical_and(inputBuffer[0][2] > 50, inputBuffer[0][2] < args.confidence_threshold)
    true_negative = np.logical_and(binary_closing(inputBuffer[5][0], disk(10)) == 255, weak_detection)
    markers[weak_detection] = 3
    """
    
    #  probable_buildings = np.logical_and(inputBuffer[0][2] > args.confidence_threshold, clean_predict == 1)
    probable_background = np.logical_and(inputBuffer[0][2] < 40, clean_predict == 0)
    ground_truth_eroded = binary_erosion(inputBuffer[5][0]==255, disk(5))
    
    probable_buildings = np.logical_and(ground_truth_eroded, inputBuffer[0][2] > 50)
    
    

    """
    ground_truth_eroded = binary_erosion(inputBuffer[5][0]==255, disk(5))
    # If WSF = 1
    probable_buildings = np.logical_and(inputBuffer[0][2] > 70, ground_truth_eroded)
    #probable_background = np.logical_and(inputBuffer[0][2] < 40, ground_truth_eroded)

    no_WSF = binary_dilation(inputBuffer[5][0]==0, disk(5)) 
    false_positive = np.logical_and(no_WSF, inputBuffer[0][2] > args.confidence_threshold)

    # note : all other pixels are 0
    markers[false_positive] = 2
    """    

    confident_buildings = np.logical_and(ground_truth_eroded, inputBuffer[0][2] > args.confidence_threshold)
    
    markers[probable_background] = 4
    markers[probable_buildings] = 1
    markers[confident_buildings] = 2
    
    
    if args.file_shadowmask:
        # shadows (note : 2 are "cleaned / big shadows", 1 is raw shadow detection)
        markers[binary_erosion(inputBuffer[4][0] == 2, disk(5))] = 8

    '''
    if args.file_vegetationmask:
        # vegetation
        markers[inputBuffer[3][0] > args.vegmask_max_value] = 7
    if args.file_watermask:
        # water
        markers[inputBuffer[2][0] == 1] = 6
    '''
         
    
    if args.remove_false_positive:
        ground_truth = inputBuffer[5][0]
        # mark as false positive pixels with high confidence but not covered by dilated ground truth
        # TODO : check if we can reduce radius for dilation
        false_positive = np.logical_and(binary_dilation(ground_truth, disk(10)) == 0, inputBuffer[0][2] > args.confidence_threshold)
        markers[false_positive] = 3
        del ground_truth, false_positive
    
    
    # watershed segmentation

    # seg[np.where(seg>3, True, False)] = 0 
    #markers[np.where(markers > 3)] = 0
    seg = segmentation.watershed(edges, markers)


    seg[np.where(seg > 3, True, False)] = 0
    seg[np.where(seg == 2, True, False)] = 1
    
    # TODO : check if we can remove/reduce this opening

    
    seg[binary_closing(seg == 1, disk(args.binary_closing))] = 1
    
    seg[binary_opening(seg == 1, disk(args.binary_closing))] = 1

    #markers[binary_opening(inputBuffer[4][0] == 2, disk(10))] = 8

    if args.remove_small_holes:
        res = remove_small_holes(
            seg.astype(bool), args.remove_small_holes, connectivity=2
        ).astype(np.uint8)
        seg = np.multiply(res, seg)

    # remove small artefacts : TODO seg contains 1, 2, 3, 4 values...
    #args.remove_small_objects = False
    if args.remove_small_objects:
        res = remove_small_objects(seg.astype(bool), args.remove_small_objects, connectivity=2).astype(np.uint8)
        # res is either 0 or 1 : we multiply by seg to keep 0/1/2 classes
        seg = np.multiply(res, seg)

    return seg, markers, edges

    

def stack(args):
    """
    Stacks all the masks given in the parameters
    :param mask: Mask path
    :param vegetation: Path of vegetation mask
    :param water: Path of water mask (classif.tif)
    :param water_pred: Path of water mask (predict.tif)
    :param building: Path of urban mask
    """

    mask_fic_veg = rio.open(args.vegetation)
    mask_veg = mask_fic_veg.read(1)
    profile = mask_fic_veg.profile

    mask_fic_water = rio.open(args.water)
    mask_water = mask_fic_water.read(1)

    mask_fic_water_pred = rio.open(args.water_pred)
    mask_water_pred = mask_fic_water_pred.read(1)

    mask_fic_urban = rio.open(args.urban)
    mask_urban = mask_fic_urban.read(1)

    mask_fic_shadow = rio.open(args.shadow)
    mask_shadow = mask_fic_shadow.read(1)

    mnh_data = np.zeros_like(mask_veg)
    height_threshold = 0
    # by default, MNH_data is set to 0 and heigth threshold is set to 0m,
    # so "mnh_data >= heigth_threshold (or <=) is always true.
    if args.mnh != "":
        mnh_fic = rio.open(args.mnh)
        mnh_data = mnh_fic.read(1)
        height_threshold = 5

    
    

    stack = np.zeros_like(mask_veg)
    height = np.zeros_like(mask_veg)
    confidence = np.zeros_like(mask_veg)

    # Urban layer : may be high, or "not confident"
    no_water_no_veg = np.logical_and(mask_water_pred == 0, mask_veg < 21)

    buildings_clean = mask_urban == 1
    urban_false_positive_clean = mask_urban == 2
    # Small holes removal
    if args.remove_small_holes:
        area = int(args.remove_small_holes)
        buildings_clean = remove_small_holes(buildings_clean.astype(bool), area, connectivity=2)
        urban_false_positive_clean = remove_small_holes(urban_false_positive_clean.astype(bool), area, connectivity=2)

    buildings = np.logical_and(
        np.logical_and(buildings_clean, mnh_data >= height_threshold),
        no_water_no_veg,
    )

    urban_false_positive = np.logical_and(urban_false_positive_clean, no_water_no_veg)

    artificial = np.logical_or(mask_urban == 2, mask_urban == 1)

    urban_areas = np.logical_and(
        np.logical_and(artificial, mnh_data <= height_threshold),
        no_water_no_veg,
    )

    stack[urban_areas] = UNDEF_URBAN_BARE_GROUND
    stack[buildings] = BUILDINGS
    stack[urban_false_positive] = BUILDINGS_FALSE_POSITIVE

    height[buildings] = HIGH
    height[urban_false_positive] = NOT_CONFIDENT
    height[urban_areas] = LOW

    confidence[buildings] += 1
    confidence[urban_false_positive] += 1

    natural_bare_ground = np.logical_and(mask_urban == 0, mask_veg == 11)

    stack[natural_bare_ground] = BARE_GROUND
    height[natural_bare_ground] = LOW

    confidence[natural_bare_ground] += 1

    low_veg = mask_veg == 21
    high_veg = np.logical_or(mask_veg == 22, mask_veg == 23)
    if args.use_mnh_veg == True:
        low_veg = np.logical_and(low_veg, mnh_data <= height_threshold)
        high_veg = np.logical_and(high_veg, mnh_data >= height_threshold)

    stack[low_veg] = LOW_VEG
    stack[high_veg] = HIGH_VEG

    height[low_veg] = LOW
    height[high_veg] = HIGH

    confidence[high_veg] += 1
    confidence[low_veg] += 1

    # Water layer : may be low (classif.tif) or not confident (predict.tif + urban)
    # Small holes removal
    if args.remove_small_holes:
        area = int(args.remove_small_holes)
        mask_water = remove_small_holes(mask_water.astype(bool), area, connectivity=2)
    water = mask_water == 1
    water_pred = np.logical_and(
        mask_veg < 11, np.logical_and(mask_water_pred == 1, mask_urban == 0)
    )
    mix_water_urban = np.logical_and(
        mask_veg < 11, np.logical_and(mask_water_pred == 1, mask_urban == 1)
    )

    # stack[mix_water_urban] = UNDEF_WATER_URBAN
    confidence[mix_water_urban] += 1
    stack[water_pred] = WATER_PRED
    stack[water] = WATER

    height[water_pred] = LOW
    height[mix_water_urban] = NOT_CONFIDENT
    height[water] = LOW

    confidence[water] += 1
    confidence[water_pred] += 1

    # Shadow
    shadow_pred = mask_shadow == 1
    # stack[shadow_pred] = SHADOW
    height[shadow_pred] = NOT_CONFIDENT

    confidence[shadow_pred] = -1

    # nb of layers
    profile["count"] = 3
    profile.update({"compress": "lzw"})
    with rio.open(args.mask, "w+", **profile) as stack_file:
        stack_file.write(stack, 1)
        stack_file.write(height, 2)
        stack_file.write(confidence, 3)
        # Add class_label in the metadata
        stack_file.update_tags(
            class_label=[
                "undef",
                "low_vegetation",
                "high_vegetation",
                "water",
                "buildings",
                "undef_water_urban",
                "bare_ground",
                "undef_urban_bare_ground",
                "water_pred",
                "shadow",
                "buildings_false_positive",
            ]
        )
    return


def watershed_regul_buildings(input_image, urban_proba, wsf, vegmask, watermask, shadowmask):
    # Compute mono image from RGB image
    im_mono = 0.29*input_image[0] + 0.58*input_image[1] + 0.114*input_image[2]
    edges = sobel(im_mono)

    markers = np.zeros((1,input_image.shape[1],input_image.shape[2]))
    
    # We set markers by reverse order of confidence
    markers[vegmask == 11] = BARE_GROUND
    
    ground_truth_eroded = binary_erosion(wsf[0]==255, disk(5)) 
    probable_buildings = np.logical_and(ground_truth_eroded, urban_proba[0] > 70)

    false_positive = np.logical_and(binary_dilation(wsf[0], disk(10)) == 0, urban_proba[0] > 70)
    
    markers[0][probable_buildings] = BUILDINGS
    markers[0][false_positive] = BUILDINGS_FALSE_POSITIVE

    
    markers[vegmask == 21] = LOW_VEG
    markers[vegmask == 22] = HIGH_VEG
    markers[vegmask == 23] = HIGH_VEG

    markers[shadowmask == 2] = BACKGROUND
    
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
    segmented_buildings, markers = watershed_regul_buildings(input_image, urban_proba, wsf, vegmask, watermask, shadowmask)

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
    stack[2] = markers
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
         help="Size of square structuring element"
     )
     
     parser.add_argument(
         "-binary_opening",
         type=int,
         default=0,
         required=False,
         action="store",
         dest="binary_opening",
         help="Size of square structuring element"
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
