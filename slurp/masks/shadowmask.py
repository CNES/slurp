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
This script computes a shadow mask
"""

import argparse
import time
import traceback
import logging
import pathlib
import json
from os import makedirs, path, remove

import eoscale.eo_executors as eoexe
import eoscale.manager as eom
import numpy as np

from slurp.post_process.morphology import apply_morpho
from slurp.tools import eoscale_utils as eo_utils
from slurp.tools import io_utils, utils
from slurp.tools.constant import NODATA_INT8

logger = logging.getLogger("slurp")


def compute_thresholds(absolute_threshold, local_phr, nodata, percentile, th_rgb, th_nir):
    """
    Compute thresholds for each band in the provided PHR image.
    If `absolute_threshold` is provided, the function will use the
    specified value as the threshold for all bands. If not, it calculates thresholds based on the
    specified percentile for each band, with different thresholds for RGB bands and NIR band.
    """
    if absolute_threshold is False:
        # Compute threshold for each band
        th_bands = np.zeros(4)
        for cpt in range(3):
            min_band = np.percentile(
                local_phr[cpt][np.where(local_phr[cpt] != nodata)],
                percentile,
            )
            max_percentile = np.percentile(
                local_phr[cpt][np.where(local_phr[cpt] != nodata)],
                100 - percentile,
            )
            th_bands[cpt] = min_band + th_rgb * (
                    max_percentile - min_band
            )

        cpt = 3
        min_band = np.percentile(
            local_phr[cpt][np.where(local_phr[cpt] != nodata)],
            percentile,
        )
        max_percentile = np.percentile(
            local_phr[cpt][np.where(local_phr[cpt] != nodata)],
            100 - percentile,
        )
        th_bands[cpt] = min_band + th_nir * (
                max_percentile - min_band
        )
    else:
        # Use an absolute threshold instead of relative threshold
        # Useful when using calibrated images
        th_bands = np.zeros(4)
        for i in range(4):
            th_bands[i] = absolute_threshold
    return th_bands


def compute_shadowmask(
    input_buffers: list, input_profiles: list, params: dict
) -> np.ndarray:
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
        raw_shadow_mask = np.logical_and(
            raw_shadow_mask, input_buffers[0][i] < params["thresholds"][i]
        )

    # Remove shadows on water areas
    raw_shadow_mask[np.where(input_buffers[2][0] == 1)] = 0

    # work on binary arrays
    final_shadow_mask = raw_shadow_mask
    if params["binary_opening"] > 0:
        final_shadow_mask = apply_morpho(
            final_shadow_mask, "binary_opening", params["binary_opening"]
        )
    if params["remove_small_objects"] > 0:
        final_shadow_mask = apply_morpho(
            final_shadow_mask,
            "remove_small_objects",
            params["remove_small_objects"],
        )

    raw_shadow_mask = np.where(raw_shadow_mask, 1, 0)
    final_shadow_mask = np.where(final_shadow_mask, 1, 0)

    # Sum between raw shadows and refined shadows
    final_shadow_mask += raw_shadow_mask

    # apply NO_DATA mask
    final_shadow_mask[np.logical_not(input_buffers[1][0])] = NODATA_INT8

    return final_shadow_mask


def getarguments() -> dict:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Compute Shadow Mask.")

    parser.add_argument(
        "main_config", help="First JSON file, load basis arguments"
    )
    parser.add_argument("-log_f",
                        "--logs_to_file",
                        action="store_true",
                        help="Store all logs to a file, instead of stdout",
                        )
    parser.add_argument("-d", "--debug", default=None, action="store_true", help="Debug flag")

    group1 = parser.add_argument_group(description="*** INPUT FILES ***")
    group1.add_argument(
        "-user_config",
        help="Second JSON file, overload basis arguments if keys are the same",
    )
    group1.add_argument("-file_vhr", help="Input 4 bands VHR image")
    group1.add_argument("-valid", dest="valid_stack", help="Validity mask")
    group1.add_argument(
        "-watermask",
        help="Watermask filename : if given shadow mask will exclude water areas",
    )

    group2 = parser.add_argument_group(description="*** OPTIONS ***")
    group2.add_argument(
        "-th_rgb", type=float, help="Relative shadow threshold for RGB bands"
    )
    group2.add_argument(
        "-th_nir", type=float, help="Relative shadow threshold for NIR band"
    )
    group2.add_argument(
        "-absolute_threshold",
        type=float,
        help="Compute shadow mask with a unique absolute threshold",
    )
    group2.add_argument(
        "-percentile",
        type=float,
        help="Percentile value to cut histogram and estimate shadow threshold",
    )

    group3 = parser.add_argument_group(description="*** POST PROCESSING ***")
    group3.add_argument(
        "-binary_opening", type=int, help="Size of disk structuring element"
    )
    group3.add_argument(
        "-remove_small_objects",
        type=int,
        help="The maximum area, in pixels, of a contiguous object that will be removed",
    )

    group4 = parser.add_argument_group(description="*** OUTPUT FILE ***")
    group4.add_argument("-shadowmask", help="Output classification filename")

    group5 = parser.add_argument_group(description="*** PARALLEL COMPUTING ***")
    group5.add_argument("-n_workers", type=int, help="Number of CPU")
    group5.add_argument(
        "-tile_max_size",
        type=int,
        help="Max tile size to be processed (0 : default)",
    )
    group5.add_argument(
        "-multiproc_context",
        default="spawn",
        help="Multiprocessing strategy: 'fork' or 'spawn' for EOScale",
    )
    args = parser.parse_args()

    arglist = []
    for arg in parser._actions:
        if arg.dest not in ["help"]:
            arglist.append(arg.dest)

    with open("args_list.json", 'w') as f:
        json.dump(arglist, f)

    return vars(args)


def slurp_shadowmask(main_config : str, logs_to_file : bool, debug: bool, user_config : str, file_vhr : str, valid_stack : bool, watermask : str, th_rgb : int,
                    th_nir : int, absolute_threshold : bool, percentile : float, binary_opening : int,
                    remove_small_objects : int, shadowmask : str, n_workers : int, tile_max_size : int, multiproc_context : str):
    """
    Main API to compute shadow mask.
    """
    t0 = time.time()

    # Read the JSON files
    keys = [
        "input",
        "aux_layers",
        "masks",
        "resources",
        "post_process",
        "shadows",
    ]
    argsdict, cli_params = utils.parse_args(keys, logs_to_file, main_config)

    for param in cli_params:
        # If the parameter from the CLI is not None, we update argsdict with the value from the CLI
        if locals()[param] is not None:
            argsdict[param] = locals()[param]

    logger.info("--" * 50)
    logger.info("SLURP - Shadow mask\n")
    logger.info(f"JSON data loaded: {main_config}")
    args = argparse.Namespace(**argsdict)

    if args.debug:
        logger.handlers[0].setLevel(logging.DEBUG) 
    logger.debug(f"{argsdict=}")   

    # Mask calculation
    with eom.EOContextManager(
            nb_workers=args.n_workers,
            tile_mode=True,
            tile_max_size=args.tile_max_size,
    ) as eoscale_manager:
        try:

            # Store image in shared memory
            key_phr = eoscale_manager.open_raster(raster_path=args.file_vhr)
            local_phr = eoscale_manager.get_array(key_phr)
            nodata = eoscale_manager.get_profile(key_phr)["nodata"]

            # Valid stack
            key_valid_stack = eoscale_manager.open_raster(
                raster_path=args.valid_stack
            )

            th_bands = compute_thresholds(args.absolute_threshold, local_phr, nodata, args.percentile, args.th_rgb, args.th_nir)

            params = {
                "thresholds": th_bands,
                "binary_opening": args.binary_opening,
                "remove_small_objects": args.remove_small_objects,
            }

            if args.watermask and path.isfile(args.watermask):
                key_watermask = eoscale_manager.open_raster(
                    raster_path=args.watermask
                )
            else:
                profile = eoscale_manager.get_profile(key_phr)
                profile["count"] = 1
                profile["dtype"] = np.uint8
                key_watermask = eoscale_manager.create_image(profile)
                eoscale_manager.get_array(key=key_watermask).fill(0)

            mask_shadow = eoexe.n_images_to_m_images_filter(
                inputs=[key_phr, key_valid_stack, key_watermask],
                image_filter=compute_shadowmask,
                filter_parameters=params,
                generate_output_profiles=eo_utils.single_uint8_profile,
                stable_margin=args.remove_small_objects,
                context_manager=eoscale_manager,
                multiproc_context=args.multiproc_context,
                filter_desc="Shadow mask processing...",
            )

            eoscale_manager.write(key=mask_shadow[0], img_path=args.shadowmask)

            end_time = time.time()
            logger.info(
                f"**** Shadow mask for {args.file_vhr} (saved as {args.shadowmask}) ****"
            )
            logger.info(
                "Total time (user)       :\t"
                + utils.convert_time(end_time - t0)
            )

        except FileNotFoundError as fnfe_exception:
            logger.error("FileNotFoundError", fnfe_exception)

        except PermissionError as pe_exception:
            logger.error("PermissionError", pe_exception)

        except ArithmeticError as ae_exception:
            logger.error("ArithmeticError", ae_exception)

        except MemoryError as me_exception:
            logger.error("MemoryError", me_exception)

        except Exception as exception:  # pylint: disable=broad-except
            logger.error("oups...", exception)
            traceback.print_exc()


def main():
    """
    Main function to run the shadow mask computation.
    It parses the command line arguments and calls the slurp_shadowmask function.
    """
    args = getarguments()
    slurp_shadowmask(**args)

if __name__ == "__main__":
    main()
