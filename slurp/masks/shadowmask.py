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
from os import makedirs, path

import eoscale.eo_executors as eoexe
import eoscale.manager as eom
import numpy as np

from slurp.post_process.morphology import apply_morpho
from slurp.tools import eoscale_utils as eo_utils
from slurp.tools import io_utils, utils
from slurp.tools.constant import NODATA_int8


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
    final_shadow_mask[np.logical_not(input_buffers[1][0])] = NODATA_int8

    return final_shadow_mask


def getarguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Compute Shadow Mask.")

    parser.add_argument(
        "main_config", help="First JSON file, load basis arguments"
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
        help="Multiprocessing strategy: 'fork' or 'spawn' for EOScale",
    )

    args = parser.parse_args()

    return args


def main():
    """Main function that compute Shadowmask"""

    argparse_dict = vars(getarguments())
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
    argsdict = io_utils.read_json(
        argparse_dict["main_config"], keys, argparse_dict.get("user_config")
    )

    # Overload with manually passed arguments if not None
    for key in argparse_dict.keys():
        if argparse_dict[key] is not None:
            argsdict[key] = argparse_dict[key]

    print("JSON data loaded:")
    print(argsdict)
    args = argparse.Namespace(**argsdict)

    # Create output folder
    makedirs(path.dirname(args.shadowmask), exist_ok=True)

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

            if args.absolute_threshold is False:
                # Compute threshold for each band
                th_bands = np.zeros(4)
                for cpt in range(3):
                    min_band = np.percentile(
                        local_phr[cpt][np.where(local_phr[cpt] != nodata)],
                        args.percentile,
                    )
                    max_percentile = np.percentile(
                        local_phr[cpt][np.where(local_phr[cpt] != nodata)],
                        100 - args.percentile,
                    )
                    th_bands[cpt] = min_band + args.th_rgb * (
                        max_percentile - min_band
                    )

                cpt = 3
                min_band = np.percentile(
                    local_phr[cpt][np.where(local_phr[cpt] != nodata)],
                    args.percentile,
                )
                max_percentile = np.percentile(
                    local_phr[cpt][np.where(local_phr[cpt] != nodata)],
                    100 - args.percentile,
                )
                th_bands[cpt] = min_band + args.th_nir * (
                    max_percentile - min_band
                )
            else:
                # Use an absolute threshold instead of relative threshold
                # Useful when using calibrated images
                th_bands = np.zeros(4)
                for i in range(4):
                    th_bands[i] = args.absolute_threshold

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
            print(
                f"**** Shadow mask for {args.file_vhr} (saved as {args.shadowmask}) ****"
            )
            print(
                "Total time (user)       :\t"
                + utils.convert_time(end_time - t0)
            )

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
