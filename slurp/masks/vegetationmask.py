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


"""Compute vegetation mask of PHR image."""

import argparse
import time
import traceback
import logging
from math import ceil, sqrt
import json
from os import makedirs, path, remove

import eoscale.eo_executors as eoexe
import eoscale.manager as eom
import numpy as np
import pandas as pd
from skimage.segmentation import slic
from sklearn.cluster import KMeans

# Cython module to compute stats
import stats as ts
from slurp.post_process.morphology import apply_morpho
from slurp.tools import eoscale_utils as eo_utils
from slurp.tools import utils
from slurp.tools.constant import NB_CLUSTERS, NODATA_INT8, NODATA_INT16

logger = logging.getLogger("slurp")

NO_VEG_CODE = 0  # Water, other non vegetated areas
UNDEFINED_VEG = 10  # Non vegetated or few vegetation (weak NDVI signal)
VEG_CODE = 20  # Vegetation

LOW_TEXTURE_CODE = 1  # Smooth areas (could be low vegetation or bare soil)
MIDDLE_TEXTURE_CODE = 2  # Middle texture areas (could be high vegetation)
HIGH_TEXTURE_CODE = 3  # High texture (could be high vegetation)

LOW_VEG_CLASS = VEG_CODE + LOW_TEXTURE_CODE
UNDEFINED_TEXTURE_CLASS = VEG_CODE + MIDDLE_TEXTURE_CODE


# MISCELLANEOUS FUNCTIONS #


def apply_map(pred, map_centroids):
    return np.array(list(map(lambda n: map_centroids[n], pred)))




# Segmentation #


def compute_segmentation(
    params: dict, img: np.ndarray, ndvi: np.ndarray
) -> np.ndarray:
    """
    Compute segmentation with SLIC

    :param dict params: dictionary of arguments
    :param np.ndarray img: input image
    :param np.ndarray ndvi: ndvi of the input image
    :returns: SLIC segments
    """
    nseg = int(img.shape[2] * img.shape[1] / params["slic_seg_size"])

    # Note : we read NDVI image.
    # Estimation of the max number of segments (ie : each segment is > 100 pixels)
    res_seg = slic(
        ndvi.astype("double"),
        compactness=float(params["slic_compactness"]),
        n_segments=nseg,
        sigma=1,
        channel_axis=None,
    )

    return res_seg


def segmentation_task(
    input_buffers: list, input_profiles: list, params: dict
) -> np.ndarray:
    """
    Segmentation

    :param list input_buffers: [im_vhr, ndvi, valid_stack]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: segments
    """
    # Note : input_buffers[x][input_buffers[2]] applies the valid mask on input_buffers[x]
    # Warning : input_buffers[0] : the mask is not applied ! But we only use NDVI mode (see compute_segmentation)
    segments = compute_segmentation(params, input_buffers[0], input_buffers[1])

    # minimum segment is 1, attribute 0 to no_data pixel
    segments[np.logical_not(input_buffers[2])] = 0
    segments[np.where(input_buffers[1] == NODATA_INT16)] = 0

    return segments


def concat_seg(previous_result, output_algo_computer, tile):
    """
    Concatenates SLIC segmentation in a single segmentation
    """
    # Computes max of previous result and adds this value to the current result :
    # prevents from computing a map with several identical labels !!
    num_seg = np.max(previous_result[0])

    previous_result[0][
        :, tile.start_y : tile.end_y + 1, tile.start_x : tile.end_x + 1
    ] = (output_algo_computer[0][:, :, :] + num_seg)

    previous_result[0][
        :, tile.start_y : tile.end_y + 1, tile.start_x : tile.end_x + 1
    ] = np.where(
        output_algo_computer[0][:, :, :] == 0,
        0,
        output_algo_computer[0][:, :, :] + num_seg,
    )


# Stats #


def compute_stats_image(
    input_buffer: list, input_profiles: list, params: dict
) -> list:
    """
    Compute the sum of each primitive and the number of pixels for each segment

    :param list input_buffer: [segments, im_ndvi, im_ndwi, im_texture]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: [ sum of each primitive ; counter (nb pixels / seg) ]
    """
    ts_stats = ts.PyStats()
    nb_primitives = len(input_buffer) - 1

    # input_buffer : list of (one band, rows, cols) images
    # [:,0,:,:] -> transform in an array (3bands, rows, cols)
    accumulator, counter = ts_stats.run_stats(
        np.array(input_buffer[1 : nb_primitives + 1])[:, 0, :, :],
        input_buffer[0],
        params["nb_lab"],
    )

    return [accumulator, counter]


def stats_concatenate(output_scalars, chunk_output_scalars, tile):
    """
    Concatenate the differents statistics on different sub-tiles parallelyzed by eoscale

    :param list output_scalars:
    :param list chunk_output_scalars:
    :param tile: bounding box of tile (not used but necessary for eoscale)
    """

    # output_scalars[0] : sums of each segment
    output_scalars[0] += chunk_output_scalars[0]
    # output_scalars[1] : counter of each segment (nb pixels/segment)
    output_scalars[1] += chunk_output_scalars[1]


# Clustering #


def apply_clustering(
    params: dict, nb_polys: int, stats: np.ndarray
) -> np.ndarray:
    """
    Apply clustering with radiometrics and texture indexes

    :param dict params: dictionary of arguments
    :param int nb_polys: number of segments
    :param np.ndarray stats: sum of each primitive for each segment
        stats[0:nb_polys] -> mean NDVI
        stats[nb_polys:2*nb_polys] -> mean NDWI
        stats[2*nb_polys:] -> mean Texture

    :returns: [ sum of each primitive ; counter (nb pixels / seg) ]
    """
    # Note : the seed for random generator is fixed to obtain reproductible results
    if params["debug"]:
        logger.debug(f"K-Means on radiometric indices {nb_polys} elements")

    kmeans_rad_indices, list_clusters, pred_veg = cluster_on_radiometry(nb_polys, params, stats)
    map_centroid, nb_clusters_no_veg, nb_clusters_veg = classify_veg_indices(kmeans_rad_indices, list_clusters, params)

    clustering = apply_map(pred_veg, map_centroid)

    # Analysis texture
    if params["texture_mode"] != "no":
        mean_texture = stats[2 * nb_polys :]
        texture_values = np.nan_to_num(
            mean_texture[np.where(clustering >= UNDEFINED_VEG)]
        )
        threshold_max = np.percentile(texture_values, params["filter_texture"])
        logger.debug("threshold_texture_max : %.2f", threshold_max)

        # Clustering
        data_textures = np.transpose(texture_values)
        data_textures[data_textures > threshold_max] = threshold_max
        if params["debug"]:
            logger.debug(
                f"K-Means on texture : {len(data_textures)} elements"
            )

        kmeans_texture = KMeans(
            n_clusters=NB_CLUSTERS,
            init="k-means++",
            n_init=5,
            verbose=0,
            random_state=712,
        )
        pred_texture = kmeans_texture.fit_predict(data_textures.reshape(-1, 1))
        
        if params["debug"]:
            logger.debug("Clustering on texture index")
            logger.debug("Clusters ordered by increasing texture values")
            logger.debug(f"\n{np.sort(kmeans_texture.cluster_centers_,axis=0)}")

        list_clusters = pd.DataFrame.from_records(
            kmeans_texture.cluster_centers_, columns=["mean_texture"]
        )
        list_clusters_by_texture = list_clusters.sort_values(
            by="mean_texture", ascending=True
        ).index

        # Attribute class
        map_centroid = []
        if params["texture_mode"] == "debug":
            # Get all clusters
            list_clusters_by_texture = list_clusters_by_texture.tolist()
            for t in range(kmeans_texture.n_clusters):
                map_centroid.append(list_clusters_by_texture.index(t))
        else:
            # Distinction veg class
            nb_clusters_high_veg = int(kmeans_texture.n_clusters / 3)
            if params["max_texture_th"]:
                # Distinction veg class by threshold
                params["nb_clusters_low_veg"] = int(
                    list_clusters[
                        list_clusters["mean_texture'"]
                        < params["max_texture_th"]
                    ].count()
                )
            if params["nb_clusters_low_veg"] >= 7:
                nb_clusters_high_veg = (
                    NB_CLUSTERS - params["nb_clusters_low_veg"]
                )
            for t in range(kmeans_texture.n_clusters):
                if (
                    t
                    in list_clusters_by_texture[: params["nb_clusters_low_veg"]]
                ):
                    map_centroid.append(LOW_TEXTURE_CODE)
                elif (
                    t
                    in list_clusters_by_texture[
                        NB_CLUSTERS - nb_clusters_high_veg :
                    ]
                ):
                    map_centroid.append(HIGH_TEXTURE_CODE)
                else:
                    map_centroid.append(MIDDLE_TEXTURE_CODE)


        textures = np.zeros(nb_polys)
        textures[np.where(clustering >= UNDEFINED_VEG)] = apply_map(
            pred_texture, map_centroid
        )

        # Ex : 10 (undefined) + 3 (textured) -> 13
        clustering = clustering + textures

    return clustering



def classify_veg_indices(kmeans_rad_indices, list_clusters, params):
    '''
    Assign vegetation class codes to each cluster based on NDVI thresholds or proportions.
    '''
    list_clusters_by_ndvi = list_clusters.sort_values(
        by="ndvi", ascending=True
    ).index
    map_centroid = []
    nb_clusters_no_veg = 0
    nb_clusters_veg = 0
    if params["min_ndvi_veg"]:
        # Attribute veg class by threshold
        for t in range(kmeans_rad_indices.n_clusters):
            if list_clusters.iloc[t]["ndvi"] > float(params["min_ndvi_veg"]):
                map_centroid.append(VEG_CODE)
                nb_clusters_veg += 1
            elif list_clusters.iloc[t]["ndvi"] < float(
                    params["max_ndvi_noveg"]
            ):
                if params["non_veg_clusters"]:
                    l_ndvi = list(list_clusters_by_ndvi)
                    v = l_ndvi.index(t)
                    map_centroid.append(v)
                else:
                    map_centroid.append(NO_VEG_CODE)  # 0
                nb_clusters_no_veg += 1
            else:
                map_centroid.append(UNDEFINED_VEG)

    else:
        # Attribute class by thirds
        nb_clusters_no_veg = int(kmeans_rad_indices.n_clusters / 3)
        if params["nb_clusters_veg"] >= 7:
            nb_clusters_no_veg = NB_CLUSTERS - params["nb_clusters_veg"]
            nb_clusters_veg = params["nb_clusters_veg"]

        for t in range(kmeans_rad_indices.n_clusters):
            if t in list_clusters_by_ndvi[:nb_clusters_no_veg]:
                if params["non_veg_clusters"]:
                    l_ndvi = list(list_clusters_by_ndvi)
                    v = l_ndvi.index(t)
                    map_centroid.append(v)
                else:
                    map_centroid.append(NO_VEG_CODE)  # 0
            elif (
                    t
                    in list_clusters_by_ndvi[
                       nb_clusters_no_veg: NB_CLUSTERS - params["nb_clusters_veg"]
                       ]
            ):
                map_centroid.append(UNDEFINED_VEG)  # 10
            else:
                map_centroid.append(VEG_CODE)  # 20
    return map_centroid, nb_clusters_no_veg, nb_clusters_veg


def cluster_on_radiometry(nb_polys, params, stats):
    '''
    K-means clustering on NDVI and NDWI indices
    '''
    kmeans_rad_indices = KMeans(
        n_clusters=NB_CLUSTERS,
        init="k-means++",
        n_init=5,
        verbose=0,
        random_state=712,
    )
    pred_veg = kmeans_rad_indices.fit_predict(
        np.stack((stats[0:nb_polys], stats[nb_polys: 2 * nb_polys]), axis=1)
    )
    if params["debug"]:
        logger.debug("Clustering on NDVI/NDWI indices")
        logger.debug("Clusters ordered by increasing NDVI values (*1000)")
        logger.debug(f"\n{np.sort(kmeans_rad_indices.cluster_centers_,axis=0)}")
    list_clusters = pd.DataFrame.from_records(
        kmeans_rad_indices.cluster_centers_, columns=["ndvi", "ndwi"]
    )
    return kmeans_rad_indices, list_clusters, pred_veg


# Finalize #


def finalize_task(input_buffers: list, input_profiles: list, params: dict):
    """
    Finalize mask : for each pixel in input segmentation, return mean NDVI

    :param list input_buffers: [segments, valid_stack]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: {"data": clusters} with clusters an array
    :returns: final mask
    """
    clustering = params["data"]
    ts_stats = ts.PyStats()

    final_mask = ts_stats.finalize(input_buffers[0], clustering)

    # Add nodata in final_mask (input_buffers[1] : valid mask)
    final_mask[np.logical_not(input_buffers[1][0])] = NODATA_INT8

    return final_mask


def clean_task(
    input_buffers: list, input_profiles: list, params: dict
) -> np.ndarray:
    """
    Post-processing : apply closing on low veg

    :param list input_buffers: [final_seg, valid_stack]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: final mask
    """
    im_classif = input_buffers[0][0]

    if params["remove_small_objects"]:
        high_veg_binary = np.where(im_classif > LOW_VEG_CLASS, True, False)
        high_veg_binary = apply_morpho(
            high_veg_binary.astype(bool),
            "remove_small_holes",
            params["remove_small_objects"],
        ).astype(np.uint8)
        im_classif[
            np.logical_and(im_classif == LOW_VEG_CLASS, high_veg_binary == 1)
        ] = UNDEFINED_TEXTURE_CLASS

    low_veg_binary = np.where(im_classif == LOW_VEG_CLASS, True, False)

    if params["remove_small_holes"]:
        low_veg_binary = apply_morpho(
            low_veg_binary.astype(bool),
            "remove_small_holes",
            params["remove_small_holes"],
        ).astype(np.uint8)
        im_classif[
            np.logical_and(im_classif > LOW_VEG_CLASS, low_veg_binary == 1)
        ] = LOW_VEG_CLASS

    if params["binary_dilation"]:
        low_veg_binary = apply_morpho(
            low_veg_binary, "binary_dilation", params["binary_dilation"]
        ).astype(np.uint8)
        im_classif[
            np.logical_and(im_classif > LOW_VEG_CLASS, low_veg_binary == 1)
        ] = LOW_VEG_CLASS

    return im_classif

def segmentation(args, eoscale_manager, key_ndvi, key_phr, key_valid_stack):
    """
    Perform image segmentation on the provided raster data (PHR, NDVI, and valid stack).
    If the save mode is set to "all" or "debug", the segmentation result
    is saved as a .tif file.

    Parameters
    ----------
    args : Namespace
        Runtime configuration and file paths.
    eoscale_manager : EOScaleManager
        The context manager responsible for managing raster I/O operations.
    key_ndvi : RasterData
        The NDVI raster data.
    key_phr : RasterData
        The PHR raster data.
    key_valid_stack : RasterData
        The valid stack raster data.
    """
    future_seg = eoexe.n_images_to_m_images_filter(
        inputs=[key_phr, key_ndvi, key_valid_stack],
        image_filter=segmentation_task,
        filter_parameters=vars(args),
        generate_output_profiles=eo_utils.single_int32_profile,
        stable_margin=0,
        context_manager=eoscale_manager,
        concatenate_filter=concat_seg,
        multiproc_context=args.multiproc_context,
        filter_desc="Segmentation processing...",
    )
    if args.save_mode in ["all", "debug"]:
        eoscale_manager.write(
            key=future_seg[0],
            img_path=args.vegetationmask.replace(".tif", "_slic.tif"),
        )
    return future_seg


def build_stack(args, eoscale_manager):
    """
    Build the required stack of input raster layers for processing.
    """
    # Image PHR
    key_phr = eoscale_manager.open_raster(raster_path=args.file_vhr)
    args.nodata_phr = eoscale_manager.get_profile(key_phr)["nodata"]
    # Valid stack
    key_valid_stack = eoscale_manager.open_raster(
        raster_path=args.valid_stack
    )
    # NDXI
    key_ndvi = eoscale_manager.open_raster(raster_path=args.file_ndvi)
    key_ndwi = eoscale_manager.open_raster(raster_path=args.file_ndwi)
    # Texture file
    key_texture = eoscale_manager.open_raster(
        raster_path=args.file_texture
    )
    return key_ndvi, key_ndwi, key_phr, key_texture, key_valid_stack


def closing(args, eoscale_manager, final_seg, key_valid_stack):
    """
    Performs morphological closing and other post-processing operations
    (binary dilation, removal of small objects, and holes,...) in the segmented image if the texture mode is enabled.

    Parameters
    ----------
    args : Namespace
        Runtime configuration and file paths.
    eoscale_manager : EOScaleManager
        The context manager responsible for managing raster I/O operations.
    final_seg : RasterData
        The segmentation result to be processed.
    key_valid_stack : RasterData
        The valid stack raster data.
    """
    if args.texture_mode == "yes" and (
            args.binary_dilation
            or args.remove_small_objects
            or args.remove_small_holes
    ):
        margin = max(
            2 * args.binary_dilation,
            ceil(sqrt(args.remove_small_objects)),
            ceil(sqrt(args.remove_small_holes)),
        )
        final_seg = eoexe.n_images_to_m_images_filter(
            inputs=[final_seg[0], key_valid_stack],
            image_filter=clean_task,
            filter_parameters=vars(args),
            generate_output_profiles=eo_utils.single_uint8_profile,
            stable_margin=margin,
            context_manager=eoscale_manager,
            multiproc_context=args.multiproc_context,
            filter_desc="Post-processing...",
        )
    return final_seg


def process_stats(args, eoscale_manager, future_seg, key_ndvi, key_ndwi, key_texture, nb_polys):
    """
    Computes statistics (mean NDVI, NDWI, and texture) for each segmented region.
    Then, the statistics are processed to generate data for clustering or classification.
    """
    params_stats = {"nb_lab": nb_polys}
    stats = eoexe.n_images_to_m_scalars(
        inputs=[future_seg[0], key_ndvi, key_ndwi, key_texture],
        image_filter=compute_stats_image,
        filter_parameters=params_stats,
        nb_output_scalars=nb_polys,
        context_manager=eoscale_manager,
        concatenate_filter=stats_concatenate,
        multiproc_context=args.multiproc_context,
        filter_desc="Stats ",
    )
    # stats[0] : sum of each primitive [ <- NDVI -><- NDWI -><- texture -> ]
    # stats[1] : nb pixels by segment   [ counter  ]
    # Once the sum of each primitive is computed, we compute the mean by dividing by the size of each segment
    np.seterr(divide="ignore", invalid="ignore")
    stats[0][:nb_polys] = stats[0][:nb_polys] / stats[1][:nb_polys]
    stats[0][nb_polys: 2 * nb_polys] = (
            stats[0][nb_polys: 2 * nb_polys] / stats[1][:nb_polys]
    )
    stats[0][2 * nb_polys: 3 * nb_polys] = (
            stats[0][2 * nb_polys: 3 * nb_polys] / stats[1][:nb_polys]
    )
    # Replace NaN by 0. After clustering, NO_DATA values will be masked
    stats[0] = np.where(np.isnan(stats[0]), 0, stats[0])
    return stats


def display_infos(args, end_time, t0, time_closing, time_cluster, time_final, time_seg, time_stack, time_stats):
    """
    Display information on the time spent on each stage of the processing pipeline.
    """
    logger.info(
        f"**** Vegetation mask for {args.file_vhr} (saved as {args.vegetationmask}) ****"
    )
    logger.info(
        "Total time (user)       :\t"
        + utils.convert_time(end_time - t0)
    )
    logger.info(
        "- Build_stack           :\t"
        + utils.convert_time(time_stack - t0)
    )
    logger.info(
        "- Segmentation          :\t"
        + utils.convert_time(time_seg - time_stack)
    )
    logger.info(
        "- Stats                 :\t"
        + utils.convert_time(time_stats - time_seg)
    )
    logger.info(
        "- Clustering            :\t"
        + utils.convert_time(time_cluster - time_stats)
    )
    logger.info(
        "- Finalize Cython       :\t"
        + utils.convert_time(time_final - time_cluster)
    )
    logger.info(
        "- Post-processing       :\t"
        + utils.convert_time(time_closing - time_final)
    )
    logger.info(
        "- Write final image     :\t"
        + utils.convert_time(end_time - time_closing)
    )
    logger.info("***")

# MAIN #


def getarguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Compute Vegetation Mask.")

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
    group1.add_argument("-ndvi", dest="file_ndvi", help="NDVI filename")
    group1.add_argument("-ndwi", dest="file_ndwi", help="NDWI filename")
    group1.add_argument(
        "-texture", dest="file_texture", help="Texture filename"
    )

    group2 = parser.add_argument_group(description="*** OPTIONS ***")
    group2.add_argument(
        "-texture_mode",
        choices=["yes", "no", "debug"],
        help=f"Labelize vegetation with (yes) or without (no) distinction low/high, "
        f"or get all {NB_CLUSTERS} vegetation clusters without distinction low/high (debug)",
    )
    group2.add_argument(
        "-filter_texture",
        type=int,
        help="Percentile for texture (between 1 and 99)",
    )
    group2.add_argument(
        "-save",
        choices=["none", "debug"],
        dest="save_mode",
        help="Save all files (debug) or only output mask (none)",
    )
    group2.add_argument(
        "-slic_seg_size", type=int, help="Approximative segment size"
    )
    group2.add_argument(
        "-slic_compactness",
        type=float,
        help="Balance between color and space proximity (see skimage.slic documentation)",
    )

    group3 = parser.add_argument_group(description="*** CLUSTERING ***")
    group3.add_argument(
        "-nb_clusters_veg",
        type=int,
        help=f"Nb of clusters considered as vegetation (1-{NB_CLUSTERS})",
    )
    group3.add_argument(
        "-min_ndvi_veg",
        type=int,
        help="Minimal mean NDVI value to consider a cluster as vegetation (overload nb clusters choice)",
    )
    group3.add_argument(
        "-max_ndvi_noveg",
        type=int,
        help="Maximal mean NDVI value to consider a cluster as non-vegetation (overload nb clusters choice)",
    )
    group3.add_argument(
        "-non_veg_clusters",
        action="store_true",
        help="Labelize each 'non vegetation cluster' as 0, 1, 2 (..) instead of single label (0)",
    )
    group3.add_argument(
        "-nb_clusters_low_veg",
        type=int,
        help=f"Nb of clusters considered as low vegetation (1-{NB_CLUSTERS})",
    )
    group3.add_argument(
        "-max_texture_th",
        type=int,
        help="Maximal texture value to consider a cluster as low vegetation (overload nb clusters choice)",
    )

    group4 = parser.add_argument_group(description="*** POST PROCESSING ***")
    group4.add_argument(
        "-binary_dilation", type=int, help="Size of disk structuring element"
    )
    group4.add_argument(
        "-remove_small_objects",
        type=int,
        help="The maximum area, in pixels, of a contiguous object that will be removed",
    )
    group4.add_argument(
        "-remove_small_holes",
        type=int,
        help="The maximum area, in pixels, of a contiguous hole that will be filled",
    )

    group5 = parser.add_argument_group(description="*** OUTPUT FILE ***")
    group5.add_argument(
        "-vegetationmask", help="Output classification filename"
    )

    group6 = parser.add_argument_group(description="*** PARALLEL COMPUTING ***")
    group6.add_argument(
        "-n_workers",
        type=int,
        help="Number of CPU for multiprocessed tasks (primitives+segmentation)",
    )
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

    arglist = []
    for arg in parser._actions:
        if arg.dest not in ["help"]:
            arglist.append(arg.dest)

    with open("args_list.json", 'w') as f:
        json.dump(arglist, f)

    return vars(args)


def slurp_vegetationmask(main_config : str, debug :bool, logs_to_file : bool, user_config : str, file_vhr : str, valid_stack : bool,
                        file_ndvi : str, file_ndwi : str, file_texture : str, texture_mode : str, filter_texture : int, save_mode : str,
                        slic_seg_size : int, slic_compactness : float, nb_clusters_veg : int, min_ndvi_veg : int,
                        max_ndvi_noveg : int, non_veg_clusters : bool, nb_clusters_low_veg : int, max_texture_th : int,
                        binary_dilation : int, remove_small_objects : int, remove_small_holes : int,
                        vegetationmask : str, n_workers : int, tile_max_size : int, multiproc_context : str):
    """
    Main API to compute shadow mask.
    """
    # Read the JSON files
    keys = [
        "input",
        "aux_layers",
        "masks",
        "resources",
        "post_process",
        "vegetation",
    ]
    argsdict, cli_params = utils.parse_args(keys, logs_to_file, main_config)

    for param in cli_params:
        # If the parameter from the CLI is not None, we update argsdict with the value from the CLI
        if locals()[param] is not None:
            argsdict[param] = locals()[param]

    logger.info("--" * 50)
    logger.info("SLURP - Vegetation mask\n")
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

            t0 = time.time()

            # Build stack with all layers #

            key_ndvi, key_ndwi, key_phr, key_texture, key_valid_stack = build_stack(args, eoscale_manager)

            time_stack = time.time()

            # Segmentation #

            future_seg = segmentation(args, eoscale_manager, key_ndvi, key_phr, key_valid_stack)

            time_seg = time.time()

            # Stats #

            # Recover number total of segments
            nb_polys = np.max(eoscale_manager.get_array(future_seg[0])[0])
            if args.debug: logger.debug(f"Number of different segments detected : {nb_polys}")
            
            # Stats calculation
            stats = process_stats(args, eoscale_manager, future_seg, key_ndvi, key_ndwi, key_texture, nb_polys)

            time_stats = time.time()

            # Clustering #

            clusters = apply_clustering(vars(args), nb_polys, stats[0])
            time_cluster = time.time()

            # Finalize mask #

            final_seg = eoexe.n_images_to_m_images_filter(
                inputs=[future_seg[0], key_valid_stack],
                image_filter=finalize_task,
                filter_parameters={"data": clusters},
                generate_output_profiles=eo_utils.single_uint8_profile,
                stable_margin=0,
                context_manager=eoscale_manager,
                multiproc_context=args.multiproc_context,
                filter_desc="Finalize processing (Cython)...",
            )

            if args.save_mode == "debug":
                eoscale_manager.write(
                    key=final_seg[0],
                    img_path=args.vegetationmask.replace(
                        ".tif", "_before_clean.tif"
                    ),
                )

            time_final = time.time()

            # Closing #

            final_seg = closing(args, eoscale_manager, final_seg, key_valid_stack)
            time_closing = time.time()

            # Write output mask #

            eoscale_manager.write(
                key=final_seg[0], img_path=args.vegetationmask
            )
            end_time = time.time()

            display_infos(args, end_time, t0, time_closing, time_cluster, time_final, time_seg, time_stack, time_stats)

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
    Main function to run the vegetation mask computation.
    It parses the command line arguments and calls the slurp_vegetationmask function.
    """
    args = getarguments()
    slurp_vegetationmask(**args)

if __name__ == "__main__":
    main()
