#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script compute all files needed for masks calculation
"""

import argparse
import numpy as np
import traceback
from os import path

from slurp.tools import io_utils, eoscale_utils as eo_utils
from slurp.prepare import validity, primitives, aux_files as aux
import eoscale.manager as eom
import eoscale.eo_executors as eoexe


def getarguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("main_config", help="First JSON file, load basis arguments")
    parser.add_argument("-user_config", help="Second JSON file, overload basis arguments if keys are the same")
    parser.add_argument("-file_vhr", help="Input 4 bands VHR image")

    # valid stack
    parser.add_argument("-valid_stack", help="Path to store the valid stack file")
    parser.add_argument("-cloud_mask", help="Path to the input cloud mask")

    # ndxi
    parser.add_argument("-file_ndvi", help="Path to store the NDVI file")
    parser.add_argument("-file_ndwi", help="Path to store the NDWI file")
    parser.add_argument("-red", type=int, help="Red band index")
    parser.add_argument("-nir", type=int, help="NIR band index")
    parser.add_argument("-green", type=int, help="Green band index")

    # Pekel and HAND
    parser.add_argument("-pekel", help="Path of the global Pekel file")
    parser.add_argument("-pekel_obs", help="Path of the global monthly has observations Pekel file")
    parser.add_argument("-extracted_pekel", help="Path to store the extracted Pekel file")
    parser.add_argument("-pekel_method",
                        help="Method for Pekel recovery : 'all' for global file and 'month' for monthly recovery")
    parser.add_argument("-hand", help="Path of the global HAND file")
    parser.add_argument("-extracted_hand", help="Path to store the extracted HAND file")

    # WSF
    parser.add_argument("-wsf", help="Path of the global WSF file")
    parser.add_argument("-extracted_wsf", help="Path to store the extracted WSF file")

    # Texture
    parser.add_argument("-file_texture", help="Path to store the texture file")
    parser.add_argument("-texture_rad", type=int, help="Radius for texture (std convolution) computation")
    
    # perfo params
    parser.add_argument("-n_workers", type=int, required=False, action="store", help="Nb of CPU")

    # only cli
    parser.add_argument("-overwrite", action="store_true", help="Recompute files even if exists")

    args = parser.parse_args()

    return args


def main():

    argparse_dict = vars(getarguments())

    # Read the JSON files
    keys = ['input', 'aux_layers', 'ressources', 'pre_process']
    argsdict = io_utils.read_json(argparse_dict["main_config"], keys, argparse_dict.get("user_config"))

    # Overload with manually passed arguments if not None
    for key in argparse_dict.keys():
        if argparse_dict[key] is not None:
            argsdict[key] = argparse_dict[key]

    args = argparse.Namespace(**argsdict)

    with eom.EOContextManager(nb_workers=args.n_workers, tile_mode=True) as eoscale_manager:
        try:
            # Store image in shared memory
            key_phr = eoscale_manager.open_raster(raster_path=args.file_vhr)
            profile = eoscale_manager.get_profile(key_phr)

            # Valid stack
            if args.overwrite or not path.isfile(args.valid_stack):
                if args.cloud_mask:
                    key_cloud_mask = eoscale_manager.open_raster(raster_path=args.cloud_mask)
                    key_valid_stack = eoexe.n_images_to_m_images_filter(
                        inputs=[key_phr, key_cloud_mask],
                        image_filter=validity.compute_valid_stack_clouds,
                        filter_parameters={"nodata": profile["nodata"]},
                        generate_output_profiles=eo_utils.single_uint8_1b_profile,
                        stable_margin=0,
                        context_manager=eoscale_manager,
                        multiproc_context="fork",
                        filter_desc="Valid stack processing..."
                    )
                else:
                    key_valid_stack = eoexe.n_images_to_m_images_filter(
                        inputs=[key_phr],
                        image_filter=validity.compute_valid_stack,
                        filter_parameters={"nodata": profile["nodata"]},
                        generate_output_profiles=eo_utils.single_uint8_1b_profile,
                        stable_margin=0,
                        context_manager=eoscale_manager,
                        multiproc_context="fork",
                        filter_desc="Valid stack processing..."
                    )
                eoscale_manager.write(key=key_valid_stack[0], img_path=args.valid_stack)
            else:
                print("Not computing valid stack mask : the file already exists.")
                key_valid_stack = [eoscale_manager.open_raster(raster_path=args.valid_stack)]

            # NDVI
            if args.overwrite or not path.isfile(args.file_ndvi):
                key_ndvi = eoexe.n_images_to_m_images_filter(
                    inputs=[key_phr, key_valid_stack[0]],
                    image_filter=primitives.compute_ndxi,
                    filter_parameters={"im_b1": args.nir, "im_b2": args.red},
                    generate_output_profiles=eo_utils.single_int16_profile,
                    stable_margin=0,
                    context_manager=eoscale_manager,
                    multiproc_context="fork",
                    filter_desc="NDVI processing..."
                )
                eoscale_manager.write(key=key_ndvi[0], img_path=args.file_ndvi)
            else:
                print("Not computing NDVI : the file already exists.")

            # NDWI
            if args.overwrite or not path.isfile(args.file_ndwi):
                key_ndwi = eoexe.n_images_to_m_images_filter(
                    inputs=[key_phr, key_valid_stack[0]],
                    image_filter=primitives.compute_ndxi,
                    filter_parameters={"im_b1": args.green, "im_b2": args.nir},
                    generate_output_profiles=eo_utils.single_int16_profile,
                    stable_margin=0,
                    context_manager=eoscale_manager,
                    multiproc_context="fork",
                    filter_desc="NDWI processing..."
                )
                eoscale_manager.write(key=key_ndwi[0], img_path=args.file_ndwi)
            else:
                print("Not computing NDWI : the file already exists.")

            # Pekel
            if args.pekel:
                if args.overwrite or not path.isfile(args.extracted_pekel):
                    if args.pekel_method == "month":
                        aux.pekel_month_recovery(args.file_vhr, args.pekel, args.extracted_pekel, args.pekel_obs)
                    elif args.pekel_method == "all":
                        aux.pekel_recovery(args.file_vhr, args.pekel, args.extracted_pekel)
                    else:
                        raise Exception("Method for Pekel extraction not accepted. Use 'month' or 'all'")
                else:
                    print("Not extracting Pekel : the file already exists.")
            else:
                print("Pass Pekel extraction")

            # Hand
            if args.hand:
                if args.overwrite or not path.isfile(args.extracted_hand):
                    aux.hand_recovery(args.file_vhr, args.hand, args.extracted_hand)
                else:
                    print("Not extracting Hand : the file already exists.")
            else:
                print("Pass Hand extraction")

            # WSF
            if args.wsf:
                if args.overwrite or not path.isfile(args.extracted_wsf):
                    aux.wsf_recovery(args.file_vhr, args.wsf, args.extracted_wsf)
                else:
                    print("Not extracting WSF : the file already exists.")
            else:
                print("Pass WSF extraction")

            # Texture
            if args.texture_rad:
                if args.overwrite or not path.isfile(args.file_texture):
                    params = {
                        "nir": args.nir,
                        "texture_rad": args.texture_rad,
                        "min_value": np.min(eoscale_manager.get_array(key_phr)[3]),
                        "max_value": np.max(eoscale_manager.get_array(key_phr)[3])
                    }
                    key_texture = eoexe.n_images_to_m_images_filter(
                        inputs=[key_phr, key_valid_stack[0]],
                        image_filter=aux.texture_task,
                        filter_parameters=params,
                        generate_output_profiles=eo_utils.single_uint16_profile,
                        stable_margin=args.texture_rad,
                        context_manager=eoscale_manager,
                        multiproc_context="fork",
                        filter_desc="Texture processing..."
                    )
                    eoscale_manager.write(key=key_texture[0], img_path=args.file_texture)
                else:
                    print("Not computing texture file : the file already exists.")
            else:
                print("Pass texture computation")

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
