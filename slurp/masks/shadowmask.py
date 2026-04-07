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
import json
import logging
import time
import traceback
from os import path
from copy import deepcopy

import numpy as np

from slurp.eomultiprocessing.slurp_executor import mp_n_to_m_images, mp_n_to_m_scalars
from slurp.eomultiprocessing.slurp_manager import slurpContextManager
from slurp.eomultiprocessing.utils import read_and_get_profile, write, read, create_image
from slurp.post_process.morphology import apply_morpho
from slurp.tools import profile_utils as eo_utils
from slurp.tools import utils
from slurp.tools.constant import NODATA_INT8

logger = logging.getLogger("slurp")


def compute_thresholds(
    absolute_threshold, local_vhr, nodata, percentile, th_rgb, th_nir
):
    """
    Compute thresholds for each band in the provided VHR image.
    If `absolute_threshold` is provided, the function will use the
    specified value as the threshold for all bands. If not, it calculates thresholds based on the
    specified percentile for each band, with different thresholds for RGB bands and NIR band.
    """
    if absolute_threshold is False:
        # Compute threshold for each bandx
        th_bands = np.zeros(4)
        for cpt in range(3):
            min_band = np.percentile(
                local_vhr[cpt][np.where(local_vhr[cpt] != nodata)],
                percentile,
            )
            max_percentile = np.percentile(
                local_vhr[cpt][np.where(local_vhr[cpt] != nodata)],
                100 - percentile,
            )
            th_bands[cpt] = min_band + th_rgb * (max_percentile - min_band)

        cpt = 3
        min_band = np.percentile(
            local_vhr[cpt][np.where(local_vhr[cpt] != nodata)],
            percentile,
        )
        max_percentile = np.percentile(
            local_vhr[cpt][np.where(local_vhr[cpt] != nodata)],
            100 - percentile,
        )
        th_bands[cpt] = min_band + th_nir * (max_percentile - min_band)
    else:
        # Use an absolute threshold instead of relative threshold
        # Useful when using calibrated images
        th_bands = np.zeros(4)
        for i in range(4):
            th_bands[i] = absolute_threshold
    return th_bands


def compute_shadowmask(
    vhr1: np.ndarray,
    vhr2: np.ndarray,
    vhr3: np.ndarray,
    vhr4: np.ndarray,
    valid_stack: np.ndarray,
    watermask: np.ndarray,
    thresholds: list,
    binary_opening: int,
    remove_small_objects: int,
) -> np.ndarray:
    """
    Compute shadow mask from spectral bands.

    Parameters
    ----------
    vhr1, vhr2, vhr3, vhr4 : np.ndarray
        Spectral bands.
    valid_stack : np.ndarray
        Validity mask.
    watermask : np.ndarray
        Water mask.
    thresholds : list[float]
        Thresholds for each band.
    binary_opening : int
    remove_small_objects : int

    Returns
    -------
    np.ndarray
        Shadow mask.
    """

    # ------------------------------
    # RAW SHADOW DETECTION
    # ------------------------------

    raw_shadow_mask = (
        (vhr1 < thresholds[0])
        & (vhr2 < thresholds[1])
        & (vhr3 < thresholds[2])
        & (vhr4 < thresholds[3])
    )

    # Remove water shadows
    raw_shadow_mask[watermask == 1] = 0

    # ------------------------------
    # MORPHOLOGY
    # ------------------------------

    final_shadow_mask = raw_shadow_mask.copy()

    if binary_opening > 0:
        final_shadow_mask = apply_morpho(
            final_shadow_mask,
            "binary_opening",
            binary_opening,
        )

    if remove_small_objects > 0:
        final_shadow_mask = apply_morpho(
            final_shadow_mask,
            "remove_small_objects",
            remove_small_objects,
        )

    # ------------------------------
    # MERGE RAW + FILTERED
    # ------------------------------

    raw_shadow_mask = raw_shadow_mask.astype(np.uint8)
    final_shadow_mask = final_shadow_mask.astype(np.uint8)

    final_shadow_mask += raw_shadow_mask

    # ------------------------------
    # APPLY NODATA
    # ------------------------------

    final_shadow_mask = np.where(
        valid_stack == 0,
        final_shadow_mask,
        NODATA_INT8,
    )

    return final_shadow_mask


def build_stack_shadow(args, slurp_manager):
    """
    Prepare inputs for shadow mask processing.

    Parameters
    ----------
    slurp_manager : object
        SLURP manager instance (unused but kept for interface consistency).

    Returns
    -------
    vhr : list
    valid_stack : list
    watermask : list
    margin : int
    vhr_profile : dict
    """

    logger.info("Loading base rasters")

    # ==============================
    # VHR
    # ==============================

    key_vhr, vhr_profile = read_and_get_profile(args.file_vhr)

    args.nodata_vhr = vhr_profile.get("nodata")
    args.shape = (vhr_profile["height"], vhr_profile["width"])
    args.crs = vhr_profile["crs"]
    args.transform = vhr_profile["transform"]

    # ==============================
    # VALID STACK
    # ==============================

    key_valid_stack = read(args.valid_stack)

    # ==============================
    # WATERMASK
    # ==============================

    if args.watermask and path.isfile(args.watermask):
        logger.info("Loading watermask")
        key_watermask = read(args.watermask)

    else:
        logger.info("No watermask provided ? creating empty mask")

        empty_profile = deepcopy(vhr_profile)
        empty_profile["count"] = 1
        empty_profile["dtype"] = np.uint8

        key_watermask = create_image(empty_profile, fill_value=0)[0]

    margin = args.remove_small_objects

    logger.info("Shadow stack ready")

    return (
        [key_vhr],
        [key_valid_stack],
        [key_watermask],
        margin,
        vhr_profile,
    )

def getarguments() -> dict:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Compute Shadow Mask.")

    parser.add_argument(
        "main_config", help="First JSON file, load basis arguments"
    )
    parser.add_argument(
        "-log_f",
        "--logs_to_file",
        action="store_true",
        help="Store all logs to a file, instead of stdout",
    )
    parser.add_argument(
        "-d", "--debug", default=None, action="store_true", help="Debug flag"
    )

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
        help="Multiprocessing strategy: 'fork' or 'spawn'",
    )
    args = parser.parse_args()

    arglist = []
    for arg in parser._actions:
        if arg.dest not in ["help"]:
            arglist.append(arg.dest)

    with open("args_list.json", "w") as f:
        json.dump(arglist, f)

    return vars(args)


def slurp_shadowmask(
    main_config: str,
    logs_to_file: bool,
    debug: bool,
    user_config: str,
    file_vhr: str,
    valid_stack: bool,
    watermask: str,
    th_rgb: int,
    th_nir: int,
    absolute_threshold: bool,
    percentile: float,
    binary_opening: int,
    remove_small_objects: int,
    shadowmask: str,
    n_workers: int,
    tile_max_size: int,
    multiproc_context: str,
):
    """
    Main API to compute shadow mask using slurpContextManager.
    """

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
        if locals()[param] is not None:
            argsdict[param] = locals()[param]

    logger.info("--" * 50)
    logger.info("SLURP - Shadow mask\n")
    logger.info(f"JSON data loaded: {main_config}")

    args = argparse.Namespace(**argsdict)

    if args.debug:
        logger.handlers[0].setLevel(logging.DEBUG)

    logger.debug(f"{argsdict=}")

    # ==============================
    # SLURP CONTEXT
    # ==============================

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
            # BUILD STACK
            # ==============================

            logger.info("[0] Step: Build stack")

            (
                vhr,
                valid_stack,
                watermask,
                margin,
                vhr_profile,
            ) = build_stack_shadow(args, slurp_manager)

            # ==============================
            # COMPUTE THRESHOLDS
            # ==============================

            logger.info("[1] Step: Compute thresholds")

            local_vhr = vhr[0]

            th_bands = compute_thresholds(
                args.absolute_threshold,
                local_vhr,
                args.nodata_vhr,
                args.percentile,
                args.th_rgb,
                args.th_nir,
            )

            params_filter = {
                "thresholds": th_bands,
                "binary_opening": args.binary_opening,
                "remove_small_objects": args.remove_small_objects,
            }

            # ==============================
            # SHADOW MASK PROCESSING
            # ==============================

            logger.info("[2] Step: Shadow mask")

            input_profile = deepcopy(vhr_profile)
            output_profile = eo_utils.single_uint8_profile(
                [deepcopy(vhr_profile)]
            )
            

            predict = mp_n_to_m_images(
                inputs=[
                    vhr[0][0],
                    vhr[0][1],
                    vhr[0][2],
                    vhr[0][3],
                    valid_stack[0][0],
                    watermask[0][0],
                ],
                image_height=input_profile["height"],
                image_width=input_profile["width"],
                output_profiles=[output_profile],
                output_keys=[path.basename(args.shadowmask)],
                func=compute_shadowmask,
                func_parameters=params_filter,
                context_manager=slurp_manager,
                stable_margin=margin,
                binary=True,
            )

            # ==============================
            # WRITE OUTPUT
            # ==============================

            slurp_manager.write_tif(
                data=predict[0],
                path=args.shadowmask,
                target_profile=output_profile,
            )

            t1 = time.time()

            logger.info(
                f"**** Shadow mask saved as {args.shadowmask} ****"
            )
            logger.info(
                "Total time (user):\t" + utils.convert_time(t1 - t0)
            )

        except Exception:
            logger.error("Unexpected error:", exc_info=True)
            traceback.print_exc()

    logger.info("End of shadowmask step\n")


def main():
    """
    Main function to run the shadow mask computation.
    It parses the command line arguments and calls the slurp_shadowmask function.
    """
    args = getarguments()
    slurp_shadowmask(**args)


if __name__ == "__main__":
    main()
