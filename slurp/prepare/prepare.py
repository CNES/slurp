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
import logging
import time
import traceback
from os import makedirs, path
from typing import List, Union
from copy import deepcopy

from slurp.eomultiprocessing.slurp_executor import mp_n_to_m_images
from slurp.eomultiprocessing.slurp_manager import slurpContextManager
from slurp.eomultiprocessing.utils import read_and_get_profile, write, read
import numpy as np

from slurp.prepare import analyse_glcm
from slurp.prepare import aux_files as aux
from slurp.prepare import geometry, primitives, validity
from slurp.tools import profile_utils as eo_utils
from slurp.tools import utils

logger = logging.getLogger("slurp")


def getarguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Compute auxiliary files needed for mask computation"
    )

    parser.add_argument(
        "main_config", help="First JSON file, load basis arguments"
    )
    parser.add_argument(
        "-d", "--debug", default=None, action="store_true", help="Debug flag"
    )
    parser.add_argument(
        "-mode",
        choices=["all", "water", "vegetation"],
        dest="mode",
        default="all",
        help="Prepare for all maks, water only or vegetation only",
    )
    parser.add_argument(
        "-w",
        "--overwrite",
        action="store_true",
        help="Recompute files even if exists",
    )
    parser.add_argument("-effective_used_config", type=str, help="")
    parser.add_argument(
        "-log_f",
        "--logs_to_file",
        action="store_true",
        help="Store all logs to a file, instead of stdout",
    )

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
        help=(
            "True if the input image is in raw (sensor) geometry, "
            "False if it is georeferenced (orthorectified)"
        ),
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
    group3.add_argument("-wbm", help="Path of the Water Body Mask (WBM) file")
    group3.add_argument(
        "-extracted_wbm", help="Path to store the extracted WBM file"
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

    # Specific case where argparse (python 3.8).
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    group5.add_argument(
        "--analyse_glcm",
        dest="analyse_glcm",
        action="store_true",
        help=(
            "Use a global land cover map to calculate the optimal number of vegetation "
            "clusters to use for mask computation"
        ),
    )
    group5.add_argument(
        "--no_analyse_glcm",
        dest="analyse_glcm",
        action="store_false",
        help="Do not analyse global land cover map",
    )
    group5.set_defaults(analyse_glcm=None)

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

    utils.store_arglist(parser)

    return vars(args)


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
    nb_clusters_veg, nb_clusters_low_veg, veg, low_veg, high_veg, non_veg = (
        analyse_glcm.compute_stats(
            args.file_vhr,
            args.land_cover_map,
            args.cropped_land_cover_map,
            args.sensor_mode,
        )
    )
    logger.info(f"veg proportion : {veg=} {low_veg=} {high_veg=} {non_veg=}")
    args_dict.update(
        {
            "nb_clusters_veg": nb_clusters_veg,
            "nb_clusters_low_veg": nb_clusters_low_veg,
            "pct_veg": veg,
            "pct_low_veg": low_veg,
            "pct_high_veg": high_veg,
            "pct_non_veg": non_veg,
        }
    )
    return args_dict


def create_valid_stack(
    args: argparse.Namespace,
    slurp_manager: slurpContextManager,
    key_vhr: str,
    profile: dict,
) -> List[str]:
    """
    Create the valid stack key using slurp executor.

    Args:
        args (Namespace): Namespace object of arguments.
        slurp_manager (slurpContextManager): slurp context manager.
        key_vhr (str): the input VHR image.
        profile (dict): raster profile.

    Returns:
        List[str]: valid_stack_key filtered output images
    """

    makedirs(path.dirname(args.valid_stack), exist_ok=True)

    # ==========================================================
    # Take the first band as it's the only one required for calc mask
    # ==========================================================
    input_keys = [key_vhr]
    
    input_profile = deepcopy(profile)
    output_profile = eo_utils.single_uint8_profile(
        [deepcopy(profile)]
    )

    if args.cloud_mask:
        key_cloud_mask = read(args.cloud_mask)
        input_keys = [key_vhr, key_cloud_mask[0]]

    valid_stack_key = mp_n_to_m_images(
        inputs=input_keys,
        image_height=input_profile["height"],
        image_width=input_profile["width"],
        output_profiles=[output_profile],
        output_keys=[path.basename(args.valid_stack)],
        func=validity.compute_valid_stack_clouds,
        func_parameters={"nodata": input_profile["nodata"]},
        context_manager=slurp_manager,
        stable_margin=0,
        binary=True,
    )

    return valid_stack_key, output_profile


def compute_ndvi(
    args: argparse.Namespace,
    slurp_manager: slurpContextManager,
    key_vhr: Union[str, np.ndarray],
    valid_stack_key: List[Union[str, np.ndarray]],
    vhr_profile: dict,
) -> List[Union[str, np.ndarray]]:
    """
    Create the Normalized Difference Vegetation Index (NDVI) layer
    using multiprocessing with slurpContextManager and return its key.

    The NDVI is computed from the input VHR image using the red
    and near-infrared (NIR) bands through the generic
    mp_n_to_m_images processing engine.

    Args:
        args (argparse.Namespace): Namespace object containing
            the processing arguments. Must include:
                - file_ndvi (str): output NDVI file path
                - red (int): index of the red band
                - nir (int): index of the NIR band
        slurp_manager (slurpContextManager): Slurp context manager
            handling multiprocessing, temporary folders, and
            output paths.
        key_vhr (Union[str, np.ndarray]): VHR input image
            (file path or numpy array).
        valid_stack_key (List[Union[str, np.ndarray]]):
            List containing the valid stack image
            (file path or numpy array).
        vhr_profile (dict): Raster profile of the VHR image.
            Must contain at least:
                - height
                - width
                - transform
                - dtype

    Returns:
        List[Union[str, np.ndarray]]:
            A list containing the NDVI image key
            (file path if disk mode, numpy array if memory mode).
    """
    # Ensure output directory exists
    makedirs(path.dirname(args.file_ndvi), exist_ok=True)
    input_profile = deepcopy(vhr_profile)
    output_profile = eo_utils.single_int16_profile(
        [deepcopy(vhr_profile)]
    )
    [ndvi_key] = mp_n_to_m_images(
        inputs=[key_vhr[args.nir - 1], key_vhr[args.red - 1], valid_stack_key[0]],
        image_height=vhr_profile["height"],
        image_width=vhr_profile["width"],
        output_profiles=[output_profile],
        output_keys=[path.basename(args.file_ndvi)],
        func=primitives.compute_ndxi,
        func_parameters={
            "im_b1": args.nir,
            "im_b2": args.red,
        },
        context_manager=slurp_manager,
        stable_margin=0,
        binary=True,
    )

    return ndvi_key, output_profile


def compute_ndwi(
    args: argparse.Namespace,
    slurp_manager: slurpContextManager,
    key_vhr: Union[str, np.ndarray],
    valid_stack_key: List[Union[str, np.ndarray]],
    vhr_profile: dict,
) -> List[Union[str, np.ndarray]]:
    """
    Create the Normalized Difference Water Index (NDWI) layer
    using multiprocessing with slurpContextManager and return its key.

    The NDWI is computed from the input VHR image using the green
    and near-infrared (NIR) bands through the generic
    mp_n_to_m_images processing engine.

    Args:
        args (argparse.Namespace): Namespace object containing
            the processing arguments. Must include:
                - file_ndwi (str): output NDWI file path
                - green (int): index of the green band
                - nir (int): index of the NIR band
        slurp_manager (slurpContextManager): Slurp context manager
            handling multiprocessing, temporary folders, and
            output paths.
        key_vhr (Union[str, np.ndarray]): VHR input image
            (file path or numpy array).
        valid_stack_key (List[Union[str, np.ndarray]]):
            List containing the valid stack image
            (file path or numpy array).
        vhr_profile (dict): Raster profile of the VHR image.
            Must contain at least:
                - height
                - width
                - transform
                - dtype

    Returns:
        List[Union[str, np.ndarray]]:
            A list containing the NDWI image key
            (file path if disk mode, numpy array if memory mode).
    """
    # Ensure output directory exists
    makedirs(path.dirname(args.file_ndwi), exist_ok=True)
    input_profile = deepcopy(vhr_profile)
    output_profile = eo_utils.single_int16_profile(
        [deepcopy(vhr_profile)]
    )
    [ndwi_key] = mp_n_to_m_images(
        inputs=[key_vhr[args.green - 1], key_vhr[args.nir - 1], valid_stack_key[0]],
        image_height=input_profile["height"],
        image_width=input_profile["width"],
        output_profiles=[output_profile],
        output_keys=[path.basename(args.file_ndwi)],
        func=primitives.compute_ndxi,
        func_parameters={
            "im_b1": args.green,
            "im_b2": args.nir,
        },
        context_manager=slurp_manager,
        stable_margin=0,
        binary=True,
    )

    return ndwi_key, output_profile


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
        if args.extracted_pekel is not None and (
            args.overwrite or not path.isfile(args.extracted_pekel)
        ):
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
            logger.info("Not extracting Pekel : the file already exists.")
    else:
        logger.info("Pass Pekel extraction")


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
        if args.extracted_hand is not None and (
            args.overwrite or not path.isfile(args.extracted_hand)
        ):
            makedirs(path.dirname(args.extracted_hand), exist_ok=True)
            aux.aux_file_recovery(
                args.file_vhr,
                args.hand,
                args.extracted_hand,
                grid_sensor,
                grid_geo,
                all_coords,
                roi,
            )
        else:
            logger.info("Not extracting Hand : the file already exists.")
    else:
        logger.info("Pass Hand extraction")


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
        if args.extracted_wsf is not None and (
            args.overwrite or not path.isfile(args.extracted_wsf)
        ):
            makedirs(path.dirname(args.extracted_wsf), exist_ok=True)
            aux.aux_file_recovery(
                args.file_vhr,
                args.wsf,
                args.extracted_wsf,
                grid_sensor,
                grid_geo,
                all_coords,
                roi,
            )
        else:
            logger.info("Not extracting WSF : the file already exists.")
    else:
        logger.info("Pass WSF extraction")


def wbm_extraction(
    args: argparse.Namespace,
    grid_sensor,
    grid_geo,
    all_coords,
    roi,
) -> None:
    """
    Extract Water Body Mask (WBM) and superimpose it on
    the VHR image.

    Args:
        args (argparse.Namespace): Namespace object of arguments.
    """

    if args.wbm and args.extracted_wbm:
        if args.extracted_wbm is not None and (
            args.overwrite or not path.isfile(args.extracted_wbm)
        ):
            makedirs(path.dirname(args.extracted_wbm), exist_ok=True)
            aux.aux_file_recovery(
                args.file_vhr,
                args.wbm,
                args.extracted_wbm,
                grid_sensor,
                grid_geo,
                all_coords,
                roi,
            )
        else:
            logger.info("Not extracting WBM : the file already exists.")
    else:
        logger.info("Pass WBM extraction")


def compute_texture(
    args: argparse.Namespace,
    slurp_manager: slurpContextManager,
    key_vhr: Union[str, np.ndarray],
    valid_stack_key: List[Union[str, np.ndarray]],
    vhr_profile: dict,
) -> List[Union[str, np.ndarray]]:
    """
    Compute texture layer used in vegetation mask using multiprocessing
    with slurpContextManager and return its key.

    Texture is computed on the NIR band using a moving window
    defined by texture_rad. Percentiles (2, 98) are calculated
    beforehand to limit the influence of outliers.

    Args:
        args (argparse.Namespace): Namespace object containing:
            - file_texture (str): output texture file path
            - nir (int): index of the NIR band
            - texture_rad (int): radius for texture computation
            - overwrite (bool): overwrite existing file
        slurp_manager (slurpContextManager): Slurp context manager
            handling multiprocessing and outputs.
        key_vhr (Union[str, np.ndarray]): VHR input image
            (file path or numpy array).
        valid_stack_key (List[Union[str, np.ndarray]]):
            List containing the valid stack image.
        vhr_profile (dict): Raster profile of the VHR image.
            Must contain at least:
                - height
                - width
                - transform
                - dtype

    Returns:
        List[Union[str, np.ndarray]]:
            A list containing the texture image key.
    """

    if not args.texture_rad:
        logger.info("Pass texture computation")
        return []

    if args.file_texture is None:
        logger.info("No texture output file provided.")
        return []

    if not args.overwrite and path.isfile(args.file_texture):
        logger.info("Not computing texture file: file already exists.")
        return []

    # Ensure output directory exists
    makedirs(path.dirname(args.file_texture), exist_ok=True)

    # ------------------------------------------------------------------
    # Compute percentiles (2, 98) on valid pixels to avoid outliers
    # ------------------------------------------------------------------
    nir_band = key_vhr[args.nir - 1]
    if isinstance(nir_band, np.ndarray):
        valid_mask = valid_stack_key[0] == 0
        percentiles = np.percentile(
            nir_band[valid_mask], [2, 98]
        )
    input_profile = deepcopy(vhr_profile)
    output_profile = eo_utils.single_uint16_profile(
        [deepcopy(vhr_profile)]
    )
    [texture_key] = mp_n_to_m_images(
        inputs=[nir_band, valid_stack_key[0]],
        image_height=input_profile["height"],
        image_width=input_profile["width"],
        output_profiles=[output_profile],
        output_keys=[path.basename(args.file_texture)],
        func=primitives.texture_task,
        func_parameters={
            "nir": args.nir,
            "texture_rad": args.texture_rad,
            "min_value": percentiles[0],
            "max_value": percentiles[1],
        },
        context_manager=slurp_manager,
        stable_margin=args.texture_rad,
        binary=False,
    )

    # write output to disk
    slurp_manager.write_tif(
        texture_key,
        args.file_texture,
        output_profile,
    )

    return [texture_key]


def valid_stack_process(
    args,
    slurp_manager,
    key_vhr,
    profile,
):
    """
    Create and save a valid stack mask if it doesn't already exist
    or if overwrite is specified.

    Uses slurpContextManager only.
    """
    if args.valid_stack is not None and (
        args.overwrite or not path.isfile(args.valid_stack)
    ):
        valid_stack_key, output_profile = create_valid_stack(
            args,
            slurp_manager,
            key_vhr,
            profile,
        )
        slurp_manager.write_tif(
            valid_stack_key[0],
            args.valid_stack,
            output_profile,
        )
    else:
        logger.info(
            "Not computing valid stack mask: file already exists."
        )

        valid_stack_key = read(args.valid_stack)

    return valid_stack_key


def sensor_mode_process(args):
    """
    Processes the VHR image based on the specified mode. If the mode is not "vegetation",
    it performs extractions for Pekel, Hand, and Water Body Mask (WBM).
    If the mode is set to "all",
    it additionally extracts the WSF (World Settlement Footprint) data.
    The extraction is done using
    the given grid and region-of-interest (ROI) computed from the image,
    DTM (Digital Terrain Model), and geoid data.
    """
    grid_sensor, grid_geo, all_coords, roi = (
        geometry.compute_interpolation_grid(
            args.file_vhr, args.dtm, args.geoid_file
        )
    )
    if args.mode != "vegetation":
        # vegetation mask doest not need external data
        # Pekel
        pekel_extraction(args, grid_sensor, grid_geo, all_coords, roi)

        # Hand
        hand_extraction(args, grid_sensor, grid_geo, all_coords, roi)

        # Water Body Mask
        wbm_extraction(args, grid_sensor, grid_geo, all_coords, roi)
    if args.mode == "all":
        # Only urban mask ('all' masks mode) need WSF
        # WSF
        wsf_extraction(args, grid_sensor, grid_geo, all_coords, roi)


def update_and_save_used_config(args_dict: dict, args: argparse.Namespace):
    """
    At the end of the prepare pipeline, update the main config.

    Args:
        args_dict (dict): dictionnary containing arguments.
        args (argparse.Namespace): args_dict instancied in Namespace object.
    """

    with open(args.main_config, "r", encoding="utf8") as json_file:
        final_used_config = json.load(json_file)
        if not isinstance(args_dict, dict):
            args_dict = vars(args_dict)
        for key in final_used_config:
            for sub_key in final_used_config[key]:
                if sub_key in args_dict:
                    final_used_config[key].update({sub_key: args_dict[sub_key]})

    makedirs(path.dirname(args.effective_used_config), exist_ok=True)
    with open(args.effective_used_config, "w", encoding="utf8") as file_to_save:
        json.dump(final_used_config, file_to_save, indent=4)


def slurp_prepare(
    main_config: str,
    debug: bool,
    mode: str,
    overwrite: bool,
    effective_used_config: str,
    logs_to_file: bool,
    user_config: str,
    file_vhr: str,
    sensor_mode: bool,
    dtm: str,
    geoid_file: str,
    valid_stack: bool,
    cloud_mask: str,
    file_ndvi: str,
    file_ndwi: str,
    red: int,
    nir: int,
    green: int,
    pekel_method: str,
    pekel: str,
    pekel_obs: str,
    pekel_monthly_occurrence: str,
    extracted_pekel: str,
    hand: str,
    extracted_hand: str,
    wsf: str,
    extracted_wsf: str,
    wbm: str,
    extracted_wbm: str,
    file_texture: str,
    texture_rad: int,
    analyse_glcm: bool,
    land_cover_map: str,
    cropped_land_cover_map: bool,
    n_workers: int,
    tile_max_size: int,
    multiproc_context: str,
):
    """
    Main function that prepares common layers (primitives, external data)
    for mask computation.
    External data recovery (Pekel, Hand, WSF) : these global raster databases
    must be superimposed on the VHR image.

    For sensor-mode only, we compute an interpolation grid taking into account
    sensor geometry, geoid and DTM.

    Uses slurpContextManager and slurp executor.
    """

    keys = ["input", "prepare", "aux_layers", "vegetation", "resources"]
    argsdict, cli_params = utils.parse_args(keys, logs_to_file, main_config)

    for param in cli_params:
        if locals()[param] is not None:
            argsdict[param] = locals()[param]

    logger.info("--" * 50)
    logger.info("SLURP - Prepare step\n")
    logger.info(f"JSON data loaded: {main_config}")
    logger.debug(argsdict)
    args = argparse.Namespace(**argsdict)
    if args.debug:
        logger.handlers[0].setLevel(logging.DEBUG)
    logger.debug(f"{argsdict=}")

    params = {
        "nb_max_workers": args.n_workers,
        "developer_mode": args.debug,
        "method": "mem",   
        "mp_context": multiproc_context,
        "output_dir": path.dirname(args.file_vhr),
    }

    with slurpContextManager(params, tile_mode=True, tile_max_size=args.tile_max_size) as slurp_manager:

        try:
            t0 = time.time()

            # ==============================
            # LOAD VHR INTO SLURP
            # ==============================
            logger.info("Step 0: Loading VHR image")
            vhr, input_profile = read_and_get_profile(args.file_vhr)

            # Global land cover map (used for vegetation mask, not water mask)
            if args.analyse_glcm and args.mode != "water":
                argsdict = add_cluster_vegetation_info(argsdict, args)

            # ==============================
            # Step 1 / 4: VALID STACK
            # ==============================
            logger.info("[1/4] Step: VALID STACK")
            valid_stack_key = valid_stack_process(
                args,
                slurp_manager,
                vhr,
                input_profile,
            )

            # ==============================
            # Step 2 / 4: NDVI
            # ==============================
            logger.info("[2/4] Step: NDVI")
            if args.file_ndvi is not None and (
                args.overwrite or not path.isfile(args.file_ndvi)
            ):
                ndvi_key, output_profile = compute_ndvi(
                    args,
                    slurp_manager,
                    vhr,
                    valid_stack_key,
                    input_profile,
                )
                slurp_manager.write_tif(
                    ndvi_key,
                    args.file_ndvi,
                    output_profile,
                )
            else:
                logger.info("NDVI skipped: file already exists.")

            # ==============================
            # Step 3 / 4: NDWI
            # ==============================
            logger.info("[3/4] Step: NDWI")
            if args.file_ndwi is not None and (
                args.overwrite or not path.isfile(args.file_ndwi)
            ):
                ndwi_key, output_profile = compute_ndwi(
                    args,
                    slurp_manager,
                    vhr,
                    valid_stack_key,
                    input_profile,
                )
                slurp_manager.write_tif(
                    ndwi_key,
                    args.file_ndwi,
                    output_profile,
                )
            else:
                logger.info("NDWI skipped: file already exists.")

            # ==============================
            # Step 4 / 4: TEXTURE (vegetation only)
            # ==============================
            if args.mode != "water":
                logger.info("[4/4] Step: TEXTURE")
                compute_texture(
                    args,
                    slurp_manager,
                    vhr,
                    valid_stack_key,
                    input_profile
                )

            # ==============================
            # SENSOR MODE
            # ==============================
            if args.sensor_mode:
                logger.info("Sensor mode processing")
                sensor_mode_process(args)

            # ==============================
            # SAVE CONFIG
            # ==============================
            update_and_save_used_config(argsdict, args)

            t1 = time.time()
            logger.info(
                "Total time (user):\t" + utils.convert_time(t1 - t0)
            )

        except Exception as exception:
            logger.error("Unexpected error:", exc_info=True)
            traceback.print_exc()

    logger.info("End of prepare step\n")


def main():
    """
    Main function to run the preparation step of SLURP.
    It parses the command line arguments and calls the slurp_prepare function.
    """
    args = getarguments()
    slurp_prepare(**args)


if __name__ == "__main__":
    main()
