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
This script compute all files needed for masks calculation
"""

import argparse
import json
import time
import traceback
from os import makedirs, path
from typing import List

import eoscale.eo_executors as eoexe
import eoscale.manager as eom
import numpy as np

from slurp.prepare import analyse_glcm
from slurp.prepare import aux_files as aux
from slurp.prepare import geometry, primitives, validity
from slurp.tools import eoscale_utils as eo_utils
from slurp.tools import io_utils, utils


def getarguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Compute auxiliary files needed for mask computation"
    )

    parser.add_argument(
        "main_config", help="First JSON file, load basis arguments"
    )
    parser.add_argument(
        "-w",
        "--overwrite",
        action="store_true",
        help="Recompute files even if exists",
    )
    parser.add_argument("-effective_used_config", type=str, help="")

    group1 = parser.add_argument_group(description="*** INPUT FILES ***")
    group1.add_argument(
        "-user_config",
        help="Second JSON file, overload basis arguments if keys are the same",
    )
    group1.add_argument("-file_vhr", help="Input 4 bands VHR image")
    group1.add_argument(
        "-sensor_mode",
        type=bool,
        default=False,
        help="True if input image is in its raw (sensor) geometry, False if input image is georeferenced (orthorectification)",
    )
    group1.add_argument(
        "-dtm", help="Digital Terrain Model, used only in sensor mode"
    )
    group1.add_argument(
        "-geoid_file", help="Geoid file, used only in sensor mode"
    )
    group1.add_argument(
        "-valid", dest="valid_stack", help="Path to store the valid stack file"
    )
    group1.add_argument("-cloud_mask", help="Path to the input cloud mask")

    group2 = parser.add_argument_group(description="*** PRIMITIVES ***")
    group2.add_argument("-file_ndvi", help="Path to store the NDVI file")
    group2.add_argument("-file_ndwi", help="Path to store the NDWI file")
    group2.add_argument("-red", type=int, help="Red band index")
    group2.add_argument("-nir", type=int, help="NIR band index")
    group2.add_argument("-green", type=int, help="Green band index")

    group3 = parser.add_argument_group(
        description="*** AUX FILES FOR WATER MASK ***"
    )
    group3.add_argument(
        "-pekel_method",
        help="Method for Pekel recovery : 'all' for global file and 'month' for monthly recovery",
    )
    group3.add_argument("-pekel", help="Path of the global Pekel file")
    group3.add_argument(
        "-pekel_obs",
        help="Month of the desired Pekel file (pekel_method = month)",
    )
    group3.add_argument(
        "-pekel_monthly_occurrence",
        help="Path of the root of monthly occurrence Pekel files",
    )
    group3.add_argument(
        "-extracted_pekel", help="Path to store the extracted Pekel file"
    )
    group3.add_argument("-hand", help="Path of the global HAND file")
    group3.add_argument(
        "-extracted_hand", help="Path to store the extracted HAND file"
    )

    group4 = parser.add_argument_group(
        description="*** AUX FILES FOR URBAN MASK ***"
    )
    group4.add_argument("-wsf", help="Path of the global WSF file")
    group4.add_argument(
        "-extracted_wsf", help="Path to store the extracted WSF file"
    )

    group5 = parser.add_argument_group(
        description="*** AUX FILES FOR VEGETATION MASK ***"
    )
    group5.add_argument("-file_texture", help="Path to store the texture file")
    group5.add_argument(
        "-texture_rad",
        type=int,
        help="Radius for texture (std convolution) computation",
    )

    # Specific case where argparse (python 3.8). https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    group5.add_argument(
        "--analyse_glcm",
        dest="analyse_glcm",
        action="store_true",
        help="Use a global land cover map to calculate the better number of vegetation cluster to use for mask computation",
    )
    group5.add_argument(
        "--no_analyse_glcm",
        dest="analyse_glcm",
        action="store_false",
        help="Do not analyse global land cover map",
    )
    group5.set_defaults(analyse_glcm=True)

    group5.add_argument(
        "-land_cover_map",
        help="Input land cover map, only used if 'analyse_glcm' is True",
    )
    group5.add_argument(
        "-cropped_land_cover_map",
        type=bool,
        help="If the land_cover_map image is cropped to the input VHR file or not",
    )

    group6 = parser.add_argument_group(description="*** PARALLEL COMPUTING ***")
    group6.add_argument("-n_workers", type=int, help="Number of CPU")
    group6.add_argument(
        "-tile_max_size",
        type=int,
        help="Max tile size to be processed (0 : default)",
    )
    group6.add_argument(
        "-multiproc_context",
        default="spawn",
        help="Multiprocessing strategy: 'fork' or 'spawn' for EOScale",
    )

    args = parser.parse_args()

    return args


def read_and_overload_arguments(args: dict) -> dict:
    """
    This function aims to read and overload arguments.
    To run the prepare pipeline, the user has to give at least 1 JSON file
    called main_config. It contains all the common parameters that the user might
    not want to modify on each run of prepare pipeline (e.g 'resources' arg).

    If the user want to overload this first JSON file or add new keys, he can do it by giving
    a second file called user_config. The user can also overload the first JSON file
    by giving other arguments that matches the keys in the main_config JSON file (e.g
    -file_vhr, etc ...).

    Args:
        arguments (dict): list of all arguments including main_config and user_config
        + all others arguments to overload.

    Returns:
        dict: The final dict of arguments.
    """
    keys_to_keep = ["input", "prepare", "aux_layers", "resources"]

    # Read the JSON files
    argsdict = io_utils.read_json(
        args["main_config"], keys_to_keep, args.get("user_config")
    )

    # Overload with manually passed arguments if not None
    for key in args.keys():
        if args[key] is not None:
            argsdict[key] = args[key]

    return argsdict


def add_cluster_vegetation_info(
    args_dict: dict, args: argparse.Namespace
) -> dict:
    """
    This function analyses a global land cover map to infer the kind of landscape
    and infer the number of vegetation clusters (resp. low vegetation).
    Then it updates the current configuration that shall be used by vegetation mask
    algorithm.

    Args:
        args_dict (dict): The parameter dictionnary used to prepare data.
        args (argparse.Namespace): Namespace object containing arguments.

    Returns:
        dict: The updated parameter dictionnary.
    """
    nb_clusters_veg, nb_clusters_low_veg = analyse_glcm.compute_stats(
        args.file_vhr,
        args.land_cover_map,
        args.cropped_land_cover_map,
        args.sensor_mode,
    )
    args_dict.update(
        {
            "nb_clusters_veg": nb_clusters_veg,
            "nb_clusters_low_veg": nb_clusters_low_veg,
        }
    )
    return args_dict


def create_valid_stack(
    args: argparse.Namespace,
    eoscale_manager: eom.EOContextManager,
    key_vhr: str,
    profile: any,
) -> List[str]:
    """
    Create the valid stack key for eoscale.
    This key is used by eoscale to access to the valid stack layer.

    Args:
        args (Namespace): Namespace object of arguments.
        eoscale_manager (eom.EOContextManager): eoscale context manager.
        key_vhr (str): key to the input VHR image.
        profile (any): raster profile.

    Returns:
        list[str]: valid_stack_key which is the
        list of VirtualPath to the filtered output images
    """
    makedirs(path.dirname(args.valid_stack), exist_ok=True)
    if args.cloud_mask:
        key_cloud_mask = eoscale_manager.open_raster(
            raster_path=args.cloud_mask
        )
        valid_stack_key = eoexe.n_images_to_m_images_filter(
            inputs=[key_vhr, key_cloud_mask],
            image_filter=validity.compute_valid_stack_clouds,
            filter_parameters={"nodata": profile["nodata"]},
            generate_output_profiles=eo_utils.single_uint8_1b_profile,
            stable_margin=0,
            context_manager=eoscale_manager,
            multiproc_context=args.multiproc_context,
            filter_desc="Valid stack processing...",
        )
    else:
        valid_stack_key = eoexe.n_images_to_m_images_filter(
            inputs=[key_vhr],
            image_filter=validity.compute_valid_stack,
            filter_parameters={"nodata": profile["nodata"]},
            generate_output_profiles=eo_utils.single_uint8_1b_profile,
            stable_margin=0,
            context_manager=eoscale_manager,
            multiproc_context=args.multiproc_context,
            filter_desc="Valid stack processing...",
        )
    return valid_stack_key


def compute_ndvi(
    args: argparse.Namespace,
    eoscale_manager: eom.EOContextManager,
    key_vhr: str,
    valid_stack_key: List[str],
) -> List[str]:
    """
    Compute the Normalized Difference Vegetation Index (ndvi) layer and
    return its key.
    This key is used by eoscale to access to the ndvi layer.

    Args:
        args (argparse.Namespace): Namespace object of arguments.
        eoscale_manager (eom.EOContextManager): eoscale context manager.
        key_vhr (str): key to the input VHR image.
        valid_stack_key (list[str]): list of VirtualPath to the
        valid stack image.

    Returns:
        list[str]: list of VirtualPath to the ndwi image.
    """
    makedirs(path.dirname(args.file_ndvi), exist_ok=True)
    key_ndvi = eoexe.n_images_to_m_images_filter(
        inputs=[key_vhr, valid_stack_key[0]],
        image_filter=primitives.compute_ndxi,
        filter_parameters={"im_b1": args.nir, "im_b2": args.red},
        generate_output_profiles=eo_utils.single_int16_profile,
        stable_margin=0,
        context_manager=eoscale_manager,
        multiproc_context=args.multiproc_context,
        filter_desc="NDVI processing...",
    )
    return key_ndvi


def compute_ndwi(
    args: argparse.Namespace,
    eoscale_manager: eom.EOContextManager,
    key_vhr: str,
    valid_stack_key: List[str],
) -> List[str]:
    """
    Create the Normalized Difference Water Index (ndwi) layer and
    return its key.

    Args:
        args (argparse.Namespace): Namespace object of arguments.
        eoscale_manager (eom.EOContextManager): eoscale context manager.
        key_vhr (str): key to the input VHR image.
        valid_stack_key (list[str]): list of VirtualPath to the
        valid stack image.

    Returns:
        list[str]: list of VirtualPath to the ndwi image.
    """
    makedirs(path.dirname(args.file_ndwi), exist_ok=True)
    key_ndwi = eoexe.n_images_to_m_images_filter(
        inputs=[key_vhr, valid_stack_key[0]],
        image_filter=primitives.compute_ndxi,
        filter_parameters={"im_b1": args.green, "im_b2": args.nir},
        generate_output_profiles=eo_utils.single_int16_profile,
        stable_margin=0,
        context_manager=eoscale_manager,
        multiproc_context=args.multiproc_context,
        filter_desc="NDWI processing...",
    )
    return key_ndwi


def pekel_extraction(
    args: argparse.Namespace, grid_sensor, grid_geo, all_coords, roi
) -> None:
    """
    Extract Global Surface Water (Pekel) and superimpose it on
    the VHR image.

    Args:
        args (argparse.Namespace): Namespace object of arguments.
    """
    if args.pekel and args.extracted_pekel:
        if args.overwrite or not path.isfile(args.extracted_pekel):
            makedirs(path.dirname(args.extracted_pekel), exist_ok=True)
            if args.pekel_method == "month":
                file_pekel = path.join(
                    args.pekel_monthly_occurrence,
                    "has_observations" + str(args.pekel_obs),
                    "has_observations" + str(args.pekel_obs) + ".vrt",
                )
                aux.aux_file_recovery(
                    args.file_vhr,
                    file_pekel,
                    args.extracted_pekel,
                    args.sensor_mode,
                    args.dtm,
                    args.geoid_file,
                    grid_sensor,
                    grid_geo,
                    all_coords,
                    roi,
                )
            elif args.pekel_method == "all":
                aux.aux_file_recovery(
                    args.file_vhr,
                    args.pekel,
                    args.extracted_pekel,
                    args.sensor_mode,
                    args.dtm,
                    args.geoid_file,
                    grid_sensor,
                    grid_geo,
                    all_coords,
                    roi,
                )
            else:
                raise Exception(
                    "Method for Pekel extraction not accepted. Use 'month' or 'all'"
                )
        else:
            print("Not extracting Pekel : the file already exists.")
    else:
        print("Pass Pekel extraction")


def hand_extraction(
    args: argparse.Namespace,
    grid_sensor,
    grid_geo,
    all_coords,
    roi,
) -> None:
    """
    Extract HAND map (Height Above Nearest Drainage) and superimpose it on
    the VHR image.

    Args:
        args (argparse.Namespace): Namespace object of arguments.
    """
    if args.hand and args.extracted_hand:
        if args.overwrite or not path.isfile(args.extracted_hand):
            makedirs(path.dirname(args.extracted_hand), exist_ok=True)
            aux.aux_file_recovery(
                args.file_vhr,
                args.hand,
                args.extracted_hand,
                args.sensor_mode,
                args.dtm,
                args.geoid_file,
                grid_sensor,
                grid_geo,
                all_coords,
                roi,
            )
        else:
            print("Not extracting Hand : the file already exists.")
    else:
        print("Pass Hand extraction")


def wsf_extraction(
    args: argparse.Namespace,
    grid_sensor,
    grid_geo,
    all_coords,
    roi,
) -> None:
    """
    Extract World Settlement Footprint (WSF) and superimpose it on
    the VHR image.

    Args:
        args (argparse.Namespace): Namespace object of arguments.
    """

    if args.wsf and args.extracted_wsf:
        if args.overwrite or not path.isfile(args.extracted_wsf):
            makedirs(path.dirname(args.extracted_wsf), exist_ok=True)
            aux.aux_file_recovery(
                args.file_vhr,
                args.wsf,
                args.extracted_wsf,
                args.sensor_mode,
                args.dtm,
                args.geoid_file,
                grid_sensor,
                grid_geo,
                all_coords,
                roi,
            )
        else:
            print("Not extracting WSF : the file already exists.")
    else:
        print("Pass WSF extraction")


def compute_texture(
    args: argparse.Namespace,
    eoscale_manager: eom.EOContextManager,
    key_vhr: str,
    valid_stack_key: List[str],
) -> None:
    """
    Compute texture used in vegetation mask

    Args:
    args (argparse.Namespace): args_dict instancied in Namespace object.
    eoscale_manager (eom.EOContextManager): eoscale context manager.
    key_vhr (str): key to the VHR input image.
    valid_stack_key (list[str]): list of VirtualPath to the
    valid stack image.
    """
    if args.texture_rad:
        if args.overwrite or not path.isfile(args.file_texture):
            makedirs(path.dirname(args.file_texture), exist_ok=True)
            # take percentiles to avoid outliers that could affect texture computation
            # compute texture on NIR band
            percentiles = np.percentile(
                eoscale_manager.get_array(key_vhr)[args.nir - 1], [2, 98]
            )
            params = {
                "nir": args.nir,
                "texture_rad": args.texture_rad,
                "min_value": percentiles[0],
                "max_value": percentiles[1],
            }
            key_texture = eoexe.n_images_to_m_images_filter(
                inputs=[key_vhr, valid_stack_key[0]],
                image_filter=aux.texture_task,
                filter_parameters=params,
                generate_output_profiles=eo_utils.single_uint16_profile,
                stable_margin=args.texture_rad,
                context_manager=eoscale_manager,
                multiproc_context=args.multiproc_context,
                filter_desc="Texture processing...",
            )
            eoscale_manager.write(
                key=key_texture[0], img_path=args.file_texture
            )
        else:
            print("Not computing texture file : the file already exists.")
    else:
        print("Pass texture computation")


def update_and_save_used_config(args_dict: dict, args: argparse.Namespace):
    """
    At the end of the prepare pipeline, update the main config.

    Args:
        args_dict (dict): dictionnary containing arguments.
        args (argparse.Namespace): args_dict instancied in Namespace object.
    """

    with open(args.main_config, "r", encoding="utf8") as json_file:
        final_used_config = json.load(json_file)
        for key in final_used_config:
            for sub_key in final_used_config[key]:
                if sub_key in args_dict:
                    final_used_config[key].update({sub_key: args_dict[sub_key]})

    makedirs(path.dirname(args.effective_used_config), exist_ok=True)
    with open(args.effective_used_config, "w", encoding="utf8") as file_to_save:
        json.dump(final_used_config, file_to_save, indent=4)


def main():
    """
    Main function that prepares common layers (primitives, external data)
    for mask computation.
    External data recovery (Pekel, Hand, WSF) : these global raster database
    must be superimposed on the VHR image.

    For sensor-mode only, we compute an interpolation grid taking into account
    sensor geometry, geoid and DTM.

    """

    argsdict = read_and_overload_arguments(vars(getarguments()))
    print("JSON data loaded:")
    print(argsdict)
    args = argparse.Namespace(**argsdict)

    # Compute prepare data with eoscale
    with eom.EOContextManager(
        nb_workers=args.n_workers,
        tile_mode=True,
        tile_max_size=args.tile_max_size,
    ) as eoscale_manager:
        try:
            t0 = time.time()

            # Store image in shared memory
            key_vhr = eoscale_manager.open_raster(raster_path=args.file_vhr)
            profile = eoscale_manager.get_profile(key_vhr)

            # Global land cover map
            if args.analyse_glcm:  # Outside of the with ?
                argsdict = add_cluster_vegetation_info(argsdict, args)

            # Valid stack
            if args.overwrite or not path.isfile(args.valid_stack):
                valid_stack_key = create_valid_stack(
                    args, eoscale_manager, key_vhr, profile
                )
                eoscale_manager.write(
                    key=valid_stack_key[0], img_path=args.valid_stack
                )
            else:
                print(
                    "Not computing valid stack mask : the file already exists."
                )
                valid_stack_key = [
                    eoscale_manager.open_raster(raster_path=args.valid_stack)
                ]

            # NDVI
            if args.overwrite or not path.isfile(args.file_ndvi):
                ndvi_key = compute_ndvi(
                    args, eoscale_manager, key_vhr, valid_stack_key
                )
                eoscale_manager.write(key=ndvi_key[0], img_path=args.file_ndvi)
            else:
                print("Not computing NDVI : the file already exists.")

            # NDWI
            if args.overwrite or not path.isfile(args.file_ndwi):
                ndwi_key = compute_ndwi(
                    args, eoscale_manager, key_vhr, valid_stack_key
                )
                eoscale_manager.write(key=ndwi_key[0], img_path=args.file_ndwi)
            else:
                print("Not computing NDWI : the file already exists.")

            if args.sensor_mode:
                grid_sensor, grid_geo, all_coords, roi = (
                    geometry.compute_interpolation_grid(
                        args.file_vhr, args.dtm, args.geoid_file
                    )
                )
            else:
                grid_sensor = None
                grid_geo = None
                all_coords = None
                roi = None

            # Pekel
            pekel_extraction(args, grid_sensor, grid_geo, all_coords, roi)

            # Hand
            hand_extraction(args, grid_sensor, grid_geo, all_coords, roi)

            # WSF
            wsf_extraction(args, grid_sensor, grid_geo, all_coords, roi)

            # Texture
            compute_texture(args, eoscale_manager, key_vhr, valid_stack_key)

            # Write effective used config
            update_and_save_used_config(argsdict, args)

            eoscale_manager._release_all()

            t1 = time.time()
            print("Total time (user)       :\t" + utils.convert_time(t1 - t0))

        except FileNotFoundError as fnfe_exception:
            print("FileNotFoundError", fnfe_exception)

        except PermissionError as pe_exception:
            print("PermissionError", pe_exception)

        except ArithmeticError as ae_exception:
            print("ArithmeticError", ae_exception)

        except MemoryError as me_exception:
            print("MemoryError", me_exception)

        except Exception as exception:
            print("oups...", exception)
            traceback.print_exc()

    print("End of prepare step")


if __name__ == "__main__":
    main()
