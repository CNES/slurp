#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script computes a shadow mask
"""

import argparse
import numpy as np
import traceback

from os import path
from skimage.morphology import binary_opening, remove_small_objects, disk

from slurp.tools import io_utils
from slurp.tools import eoscale_utils as eo_utils
import eoscale.manager as eom
import eoscale.eo_executors as eoexe

NO_DATA = 255


def compute_mask(input_buffers: list, input_profiles: list, params: dict) -> np.ndarray:
    """
    Compute shadow mask

    :param list input_buffers: 0 -> image, 1 -> valid_stack, 2 -> watermask
    :param list input_profiles: image profiles (not used but necessary for eoscale)
    :param dict params: must contain the keys "thresholds", "binary_opening" and "small_objects"
    :returns: valid_phr (boolean numpy array, True = valid data, False = no data)
    """
    raw_shadow_mask = np.zeros(input_buffers[0][0].shape, dtype=int)
    raw_shadow_mask.fill(1)
    
    for i in range(4):
        raw_shadow_mask = np.logical_and(raw_shadow_mask, input_buffers[0][i] < params["thresholds"][i])

    # Remove shadows on water areas
    raw_shadow_mask[np.where(input_buffers[2][0] == 1)] = 0
        
    # work on binary arrays
    final_shadow_mask = raw_shadow_mask
    if params["binary_opening"] > 0:
        final_shadow_mask = binary_opening(raw_shadow_mask, disk(params["binary_opening"]))
    if params["remove_small_objects"] > 0:
        final_shadow_mask = remove_small_objects(final_shadow_mask, params["remove_small_objects"], connectivity=2)
        
    raw_shadow_mask = np.where(raw_shadow_mask, 1, 0)
    final_shadow_mask = np.where(final_shadow_mask, 1, 0)

    # Sum between raw shadows and refined shadows
    final_shadow_mask += raw_shadow_mask

    # apply NO_DATA mask
    final_shadow_mask[np.logical_not(input_buffers[1][0])] = NO_DATA
    
    return final_shadow_mask


def getarguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("main_config", help="First JSON file, load basis arguments")
    parser.add_argument("-user_config", help="Second JSON file, overload basis arguments if keys are the same")
    parser.add_argument("-file_vhr", help="Input 4 bands VHR image")
    parser.add_argument("-shadowmask", help="Final mask")
    
    # aux files
    parser.add_argument("-valid_stack", help="Validity mask")

    # computation params
    parser.add_argument("-th_rgb", type=float, action="store",
                        help="Relative shadow threshold for RGB bands (default 0.3)")
    parser.add_argument("-th_nir", type=float, action="store",
                        help="Relative shadow threshold for NIR band (default 0.3)")
    parser.add_argument("-absolute_threshold", type=float, required=False, action="store",
                        help="Compute shadow mask with a unique absolute threshold")
    parser.add_argument("-percentile", help="Percentile value to cut histogram and estimate shadow threshold")
    parser.add_argument("-binary_opening", "--binary_opening", type=int, required=False, action="store",
                        help="Size of ball structuring element")
    parser.add_argument("-remove_small_objects", "--remove_small_objects", type=int, required=False, action="store",
                        help="The maximum area, in pixels, of a contiguous object that will be removed")
    parser.add_argument("-watermask", required=False, action="store", dest="watermask",
                        help="Watermask filename : shadow mask will exclude water areas")

    # perfo params
    parser.add_argument("-n_workers", type=int, required=False, action="store", help="Nb of CPU")

    args = parser.parse_args()

    return args


def main():
   
    argparse_dict = vars(getarguments())

    # Read the JSON files
    keys = ['input', 'aux_layers', 'masks', 'ressources', 'shadows']
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

            # Store image in shared memory
            key_phr = eoscale_manager.open_raster(raster_path=args.file_vhr)
            local_phr = eoscale_manager.get_array(key_phr)
            nodata = eoscale_manager.get_profile(key_phr)["nodata"]

            # Valid stack
            key_valid_stack = eoscale_manager.open_raster(raster_path=args.valid_stack)

            if args.absolute_threshold is False:
                # Compute threshold for each band
                th_bands = np.zeros(4)
                for cpt in range(3):
                    min_band = np.percentile(local_phr[cpt][np.where(local_phr[cpt] != nodata)], args.percentile)
                    max_percentile = np.percentile(local_phr[cpt][np.where(local_phr[cpt] != nodata)], 100-args.percentile)
                    th_bands[cpt]  = min_band + args.th_rgb * (max_percentile - min_band)
                    
                cpt = 3
                min_band = np.percentile(local_phr[cpt][np.where(local_phr[cpt] != nodata)], args.percentile)
                max_percentile = np.percentile(local_phr[cpt][np.where(local_phr[cpt] != nodata)], 100-args.percentile)
                th_bands[cpt]  = min_band + args.th_nir * (max_percentile - min_band)
            else:
                # Use an absolute threshold instead of relative threshold
                # Useful when using calibrated images
                th_bands = np.zeros(4)
                for i in range(4):
                    th_bands[i] = args.absolute_threshold
                    
            params = {
                "thresholds": th_bands,
                "binary_opening": args.binary_opening,
                "remove_small_objects": args.remove_small_objects
            }

            if args.watermask and path.isfile(args.watermask):
                key_watermask = eoscale_manager.open_raster(raster_path=args.watermask)
            else:
                profile = eoscale_manager.get_profile(key_phr)
                profile["count"] = 1
                profile["dtype"] = np.uint8
                key_watermask = eoscale_manager.create_image(profile)
                eoscale_manager.get_array(key=key_watermask).fill(0)
            
            mask_shadow = eoexe.n_images_to_m_images_filter(inputs=[key_phr, key_valid_stack, key_watermask],
                                                            image_filter=compute_mask,
                                                            filter_parameters=params,
                                                            generate_output_profiles=eo_utils.single_uint8_profile,
                                                            stable_margin=args.remove_small_objects,
                                                            context_manager=eoscale_manager,
                                                            filter_desc="Shadow mask processing...")

            eoscale_manager.write(key=mask_shadow[0], img_path=args.shadowmask)

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
