#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script compute all files needed for masks calculation
"""

import argparse
import rasterio as rio
import traceback
from os import path

from slurp.tools import io_utils, eoscale_utils as eo_utils
from slurp.prepare import validity
import eoscale.manager as eom
import eoscale.eo_executors as eoexe


def getarguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("main_config", help="First JSON file, load basis arguments")
    parser.add_argument("-user_config", help="Second JSON file, overload basis arguments if keys are the same")
    parser.add_argument("-file_vhr", help="Input 4 bands VHR image")

    # valid stack
    parser.add_argument("-valid_stack", help="Path to store valid stack file")
    
    # perfo params
    parser.add_argument("-n_workers", type=int, required=False, action="store", help="Nb of CPU")

    # only cli
    parser.add_argument("-overwrite", action="store_true", help="Recompute files even if exists")

    args = parser.parse_args()

    return args


def main():

    argparse_dict = vars(getarguments())

    # Read the JSON files
    keys = ['input', 'aux_layers', 'ressources']
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

            # Valid stack
            if args.overwrite or not path.isfile(args.valid_stack):
                ds_phr = rio.open(args.file_vhr)
                nodata = ds_phr.profile["nodata"]
                ds_phr.close()

                key_valid_stack = eoexe.n_images_to_m_images_filter(
                    inputs=[key_phr],
                    image_filter=validity.compute_valid_stack,
                    filter_parameters={"nodata": nodata},
                    generate_output_profiles=eo_utils.single_bool_profile,
                    stable_margin=0,
                    context_manager=eoscale_manager,
                    multiproc_context="fork",
                    filter_desc="Valid stack processing..."
                )

                eoscale_manager.write(key=key_valid_stack[0], img_path=args.valid_stack)
            else:
                print("Not computing valid stack mask : the file already exists.")

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
