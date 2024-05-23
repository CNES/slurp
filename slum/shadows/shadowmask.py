#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script stacks existing masks
"""

import argparse
import rasterio
import numpy as np
import argparse
import os
import traceback

from skimage.morphology import binary_closing, binary_opening, binary_erosion, remove_small_objects, disk, remove_small_holes


import eoscale.manager as eom
import eoscale.eo_executors as eoexe

NO_DATA=255

def compute_mask(input_buffers: list, 
                 input_profiles: list, 
                 params: dict) -> np.ndarray :
    
    """
    input_buffers : 
    0 -> image
    1 -> valid_stack
    """
    
    raw_shadow_mask = np.zeros(input_buffers[0][0].shape, dtype=int)
    raw_shadow_mask.fill(1)
    
    for i in range(4):
        raw_shadow_mask = np.logical_and(raw_shadow_mask, input_buffers[0][i] < params["thresholds"][i])
    
    # work on binary arrays
    final_shadow_mask = raw_shadow_mask
    if params["binary_opening"] > 0:
        print(f"binary opening : {params['binary_opening']=}")
        final_shadow_mask = binary_opening(raw_shadow_mask, disk(params["binary_opening"]))
    if params["small_objects"] > 0:
        final_shadow_mask = remove_small_objects(final_shadow_mask, params["small_objects"], connectivity=2)
        
    raw_shadow_mask = np.where(raw_shadow_mask,1,0)
    final_shadow_mask = np.where(final_shadow_mask,1,0)
    
    final_shadow_mask += raw_shadow_mask

    final_shadow_mask[np.logical_not(input_buffers[1][0])] = NO_DATA
    
    return final_shadow_mask

def single_uint8_profile(input_profiles: list, map_params):
    profile = input_profiles[0]
    profile['count']=1
    profile['dtype']=np.uint8
    profile["compress"] = "lzw"
    profile["nodata"] = NO_DATA
    
    return profile

def single_bool_profile(input_profiles: list, map_params):
    profile = input_profiles[0]
    profile['count']=1
    profile['dtype']=bool
    profile["compress"] = "lzw"
    
    return profile

def compute_valid_stack(inputBuffer: list, 
            input_profiles: list, 
            args: dict) -> list:
    #inputBuffer = [im_vhr]
    # Valid_phr (boolean numpy array, True = valid data, False = no data)
    valid_phr = np.logical_and.reduce(inputBuffer[0] != args["nodata"], axis=0)
        
    return valid_phr

def getarguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Input 4 bands VHR image")
    parser.add_argument("mask", help="Final mask")
    parser.add_argument("-th_rgb", default=0.3, type=float, action="store", help="Relative shadow threshold for RGB bands (default 0.3)")
    parser.add_argument("-th_nir", default=0.3, type=float, action="store", help="Relative shadow threshold for NIR band (default 0.3)")
    parser.add_argument("-percentile", default=2, help="Percentile value to cut histogram and estimate shadow threshold")
    parser.add_argument("-binary_opening","--binary_opening", type=int, required=False, default=0, action="store",
                        help="Size of ball structuring element")
    parser.add_argument("-remove_small_objects","--small_objects", type=int, required=False, default=0, action="store",
                        help="The maximum area, in pixels, of a contiguous object that will be removed")
 
    parser.add_argument("-n_workers",type=int, default=8, required=False, action="store", dest="nb_workers", help="Nb of CPU" )

    args = parser.parse_args()

    return args

def main():
    args = getarguments()
    
    with eom.EOContextManager(nb_workers = args.nb_workers, tile_mode = True) as eoscale_manager:
        try:

            # Store image in shared memmory
            key_phr = eoscale_manager.open_raster(raster_path = args.image)
            local_phr = eoscale_manager.get_array(key_phr)

            ds_phr = rasterio.open(args.image)
            nodata = ds_phr.profile["nodata"]
            
            # Compute threshold for each band
            th_bands = np.zeros(4)
            for cpt in range(3):
                min_band = np.percentile(local_phr[cpt][np.where(local_phr[cpt]!=nodata)],args.percentile)
                max_percentile = np.percentile(local_phr[cpt][np.where(local_phr[cpt]!=nodata)], 100-args.percentile)
                th_bands[cpt]  = min_band + args.th_rgb * (max_percentile - min_band)
            

            cpt = 3
            min_nir = np.percentile(local_phr[cpt][np.where(local_phr[cpt]!=nodata)],args.percentile)
            max_percentile = np.percentile(local_phr[cpt][np.where(local_phr[cpt]!=nodata)], 100-args.percentile)
            th_bands[cpt]  = min_band + args.th_nir * (max_percentile - min_band)
            
            params = {"thresholds":th_bands, "binary_opening":args.binary_opening, "small_objects":args.small_objects, "nodata":nodata}

            key_valid_stack = eoexe.n_images_to_m_images_filter(inputs = [key_phr],
                                                                image_filter = compute_valid_stack,   
                                                                filter_parameters=params,
                                                                generate_output_profiles = single_bool_profile,
                                                                stable_margin= 0,
                                                                context_manager = eoscale_manager,
                                                                multiproc_context= "fork",
                                                                filter_desc= "Valid stack processing...")
            
            mask_shadow = eoexe.n_images_to_m_images_filter(inputs = [key_phr, key_valid_stack[0]],
                                                           image_filter = compute_mask,
                                                           filter_parameters=params,
                                                           generate_output_profiles = single_uint8_profile,
                                                           stable_margin= args.small_objects,
                                                           context_manager = eoscale_manager,
                                                           filter_desc= "Shadow mask processing...")          

            eoscale_manager.write(key = mask_shadow[0], img_path = args.mask)

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
    
    
