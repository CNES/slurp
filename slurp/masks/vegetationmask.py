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


"""Compute vegetation mask of VHR image."""

import argparse
import json
import logging
import time
import traceback
from os import path
from copy import deepcopy
from math import ceil, sqrt

import numpy as np
from skimage.segmentation import slic
from sklearn.cluster import KMeans

from slurp.eomultiprocessing.slurp_executor import mp_n_to_m_images, mp_n_to_m_scalars, mp_n_to_m_images_with_mapping
from slurp.eomultiprocessing.slurp_manager import slurpContextManager
from slurp.eomultiprocessing.utils import read_and_get_profile, write, read

# Cython module to compute stats
import stats as ts
from slurp.post_process.morphology import apply_morpho
from slurp.tools import profile_utils as eo_utils
from slurp.tools import random_forest_utils, utils
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
    return np.array([map_centroids[n] for n in pred])




def build_stack_vegetation(args, slurp_manager):
    """
    Prepare image layers required for vegetation mask processing
    using slurpContextManager.

    Parameters
    ----------
    args : Namespace
        Expected attributes:
            - file_vhr : str
            - valid_stack : str
            - file_ndvi : str
            - file_ndwi : str
            - file_texture : str

        Updated in-place:
            - nodata_vhr
            - shape
            - crs
            - transform
            - rpc (set to None)

    slurp_manager : slurpContextManager
        SLURP context manager handling raster access.

    Returns
    -------
    key_ndvi : list
    key_ndwi : list
    key_vhr : list
    key_texture : list
    key_valid_stack : list
    margin : int
    profile_vhr : dict
    profile_texture : dict
    """

    # ======================================================
    # VHR IMAGE
    # ======================================================

    key_vhr, profile_vhr = read_and_get_profile(args.file_vhr)

    args.nodata_vhr = profile_vhr.get("nodata")
    args.shape = (profile_vhr["height"], profile_vhr["width"])
    args.crs = profile_vhr["crs"]
    args.transform = profile_vhr["transform"]
    args.rpc = None  # Not handled in SLURP mode

    # ======================================================
    # VALID STACK
    # ======================================================

    key_valid_stack = read(args.valid_stack)

    # ======================================================
    # NDVI / NDWI
    # ======================================================

    key_ndvi, profile_ndvi = read_and_get_profile(args.file_ndvi)
    key_ndwi = read(args.file_ndwi)

    # ======================================================
    # TEXTURE LAYER
    # ======================================================

    key_texture, profile_texture = read_and_get_profile(
        args.file_texture
    )


    # ======================================================
    # RETURN (SLURP FORMAT)
    # ======================================================

    return (
        [key_ndvi],
        [key_ndwi],
        [key_vhr],
        [key_texture],
        [key_valid_stack],
        profile_vhr,
        profile_texture,
        profile_ndvi
    )

# Segmentation #

def compute_segmentation(params: dict, ndvi: np.ndarray) -> np.ndarray:
    """
    Compute segmentation with SLIC

    :param dict params: dictionary of arguments
    :param np.ndarray ndvi: ndvi of the input image
    :returns: SLIC segments
    """
    # computes nb of expected segments taking account NO_DATA
    """
    nseg = int(len(ndvi[ndvi!=NODATA_INT16]) / params["slic_seg_size"])
    if nseg == 0:
        logger.debug(f"Taille de segments : 0 !!  attention, risque de div par zero {ndvi.shape=}")
    TODO : clean / choose right way. 
    print(f"Nseg - X*Y / slic_seg_size (do not take into account NODATA) : {nseg}\n"
          f"Alternative count : {int(len(ndvi[ndvi!=NODATA_INT16]) / params['slic_seg_size'])=}")

    """
    # nseg = int(ndvi.shape[2] * ndvi.shape[1] / params["slic_seg_size"])
    # nseg cannot be equal to 0, because calling function already checked
    # there were valid pixels to segment...
    nseg = int(len(ndvi[ndvi != NODATA_INT16]) / params["slic_seg_size"])

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
    ndvi: np.ndarray,
    valid_stack: np.ndarray,
    slic_seg_size: int,
    slic_compactness: float,
) -> np.ndarray:
    """
    Segments NDVI with SLIC algorithm and masks invalid pixels.

    Parameters
    ----------
    ndvi : np.ndarray
        NDVI image tile
    valid_stack : np.ndarray
        Validity mask (0 = valid pixel)
    slic_seg_size : int
        Target SLIC segment size
    slic_compactness : float
        SLIC compactness parameter

    Returns
    -------
    np.ndarray
        Segmentation labels
    """

    # count valid pixels
    nb_val_zero = len(np.where(valid_stack == 0)[0])

    if nb_val_zero == 0:
        return np.zeros_like(valid_stack)

    params = {
        "slic_seg_size": slic_seg_size,
        "slic_compactness": slic_compactness,
    }
    segments = compute_segmentation(params, ndvi)

    # mask invalid pixels
    segments = np.where(valid_stack == 0, segments, 0)

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

    # TODO : check if we can keep only this statement
    previous_result[0][
        :, tile.start_y : tile.end_y + 1, tile.start_x : tile.end_x + 1
    ] = np.where(
        output_algo_computer[0][:, :, :] == 0,
        0,
        output_algo_computer[0][:, :, :] + num_seg,
    )


# Stats #


def compute_stats_image(
    segments: np.ndarray,
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    texture: np.ndarray,
    nb_lab: int
) -> list:
    """
    Compute the sum of each primitive and the number of pixels for each segment.

    Parameters
    ----------
    segments : np.ndarray
        Segmentation labels
    ndvi : np.ndarray
        NDVI tile (H,W)
    ndwi : np.ndarray
        NDWI tile (H,W)
    texture : np.ndarray
        Texture tile (H,W)
    nb_lab : int
        Number of labels in segments

    Returns
    -------
    list
        [accumulator (sum per segment), counter (number of pixels per segment)]
    """

    ts_stats = ts.PyStats()
    nb_primitives = 3  # NDVI, NDWI, Texture

    # Normalize inputs in case they come as (1,H,W)
    def ensure_2d(arr):
        if arr.ndim == 3 and arr.shape[0] == 1:
            return arr[0]
        return arr

    ndvi = ensure_2d(ndvi)
    ndwi = ensure_2d(ndwi)
    texture = ensure_2d(texture)
    segments = ensure_2d(segments)

    # stack primitives as (nb_primitives, H, W)
    primitives_stack = np.stack([ndvi, ndwi, texture], axis=0)

    accumulator, counter = ts_stats.run_stats(
        primitives_stack,
        segments,
        nb_lab,
    )

    return [accumulator, counter] 


def stats_concatenate(chunks_output_scalars):
    """
    Concatenate statistics coming from multiple sub-tiles parallelized by eoscale.

    Each entry of chunks_output_scalars is:
        [sum_array, count_array]

    :param list chunks_output_scalars: list of chunk statistics
    :return: [global_sum, global_count]
    """

    # Init with first chunk
    global_sum = np.array(chunks_output_scalars[0][0], copy=True)
    global_count = np.array(chunks_output_scalars[0][1], copy=True)

    # Concatenate other chunks
    for chunk_sum, chunk_count in chunks_output_scalars[1:]:
        global_sum += chunk_sum
        global_count += chunk_count

    return [global_sum, global_count]


def clustering_vegetation(
    params: dict,
    size_result: int,
    stats: np.ndarray,
    mask_valid_indices: np.ndarray,
) -> np.ndarray:
    """
    Classify segments with a k-means clustering, based on NDVI/NDWI values
    returns a list of segments with their cluster index (0..nb_clusters),
    ordered by increasing mean NDVI value
    :param dict params: arguments of the algorithm
    :param int size_result: number of segment detected
    :param np.ndarray stats: sum of each primitive for each segment
        stats[0:size_result] -> mean NDVI
        stats[size_result:2*size_result] -> mean NDWI
        stats[2*size_result:] -> mean Texture
    :param np.ndarray mask_valid_indices: mask of valid segments indices

    returns [ array of segments, with their cluster index (ordered by NDVI value),
    list of NDVI centroids values (used by one labeling method) ]
    """
    kmeans_rad_indices = KMeans(
        n_clusters=NB_CLUSTERS,
        init="k-means++",
        n_init=5,
        verbose=0,
        random_state=712,
    )

    ndvi = stats[0:size_result]
    ndwi = stats[size_result : 2 * size_result]
    logger.debug(f"Before NODATA removal {ndvi.shape=}")
    ndvi = ndvi[np.where(mask_valid_indices)]
    ndwi = ndwi[np.where(mask_valid_indices)]
    vec_predic = np.stack((ndvi, ndwi), axis=1)
    logger.debug(
        f"{len(np.where(mask_valid_indices)[0])=} -> after NODATA removal {ndvi.shape=}"
    )

    pred_veg = kmeans_rad_indices.fit_predict(vec_predic)

    ndvi_values = [v[0] for v in kmeans_rad_indices.cluster_centers_]
    sorted_ndvi = np.sort(ndvi_values).tolist()

    sorted_clusters = np.array([sorted_ndvi.index(v) for v in ndvi_values])
    logger.debug(
        f"1st clustering : NDVI centroids : {sorted_ndvi} {sorted_clusters=}"
    )
    pred_veg_sorted = apply_map(pred_veg, sorted_clusters)

    return pred_veg_sorted, sorted_ndvi


def clustering_texture(
    params: dict,
    size_result: int,
    stats: np.ndarray,
    clustering: np.ndarray,
    mask_valid_indices: np.ndarray,
) -> np.ndarray:
    """
    Classify segments with a k-means clustering, based on texture value.
    Values are normalized before k-means step.
    returns a list of segments with their cluster index (0..nb_clusters),
    ordered by increasing mean texture value
    :param dict params: arguments of the algorithm
    :param int size_result: size of the final result
                            (nb of segments initially detected + 1 for NODATA areas)
    :param np.ndarray stats: sum of each primitive for each segment
        stats[0:size_result] -> mean NDVI
        stats[size_result:2*size_result] -> mean NDWI
        stats[2*size_result:] -> mean Texture
    :param np.ndarray mask_valid_indices: mask of valid segments indices

    returns [ array of segments, with their cluster index (ordered by texture value),
    list of texture centroid values]
    """

    mean_texture = stats[2 * size_result :]
    """
    TODO : check if we can remove this
    texture_values = np.nan_to_num(
        mean_texture[np.where(clustering >= UNDEFINED_VEG)]
    )
    """
    veg_values = mean_texture[np.where(mask_valid_indices)]
    texture_values = veg_values[np.where(clustering >= UNDEFINED_VEG)]

    threshold_max = np.percentile(texture_values, params["filter_texture"])
    data_textures = np.transpose(texture_values)
    data_textures[data_textures > threshold_max] = threshold_max

    kmeans_texture = KMeans(
        n_clusters=NB_CLUSTERS,
        init="k-means++",
        n_init=5,
        verbose=0,
        random_state=712,
    )
    pred_texture = kmeans_texture.fit_predict(data_textures.reshape(-1, 1))

    texture_values = [v[0] for v in kmeans_texture.cluster_centers_]
    sorted_texture = np.sort(texture_values).tolist()

    sorted_clusters = np.array(
        [sorted_texture.index(v) for v in texture_values]
    )
    logger.debug(f"2nd clustering : Texture centroids : {sorted_texture}")
    textures = np.zeros(size_result).astype(np.uint8)
    textures[np.where(clustering >= UNDEFINED_VEG)] = apply_map(
        pred_texture, sorted_clusters
    )
    # textures = [ 0  0  0    8 8 8 7 8 7   1 3 2 3  1 1 .. ]
    #              (nonveg)  (textured veg)  (smooth veg)
    return textures, sorted_clusters


def frac_veg_from_segments(segments, params: dict):
    """
    Estimate number of vegetation and non-vegetation clusters from a target ratio
    and the repartition of areas in the previous clustering step.
    Ratio can come from a global LandCover Map (ie : ESA WorldCover)

    To improve computation time, ratio of areas are estimated by counting
    segments (superpixels) instead of computing exact areas. SLIC segmentation
    produces quite homogeneous segments so this is quite acceptable.
    """

    nb_segments = segments.shape[0]
    # for each index of cluster from 8 (NB_CLUSTERS) to 0, compute ratio of segments over this index
    ratios_surfaces = [
        np.where(segments >= i)[0].shape[0] / nb_segments
        for i in range(NB_CLUSTERS - 1, -1, -1)
    ]

    if params["labeling_strategy"] == "nearest":
        takeClosest = lambda num, collection: collection.index(
            min(collection, key=lambda x: abs(x - num))
        )
        index_cluster_veg = takeClosest(params["pct_veg"], ratios_surfaces)

    else:
        # lists of cluster that overestimate (resp underestimate) the vegetation ratio
        clusters_over, clusters_under = [], []
        for x in ratios_surfaces:
            if x - params["pct_veg"] > 0:
                clusters_over.append(x)
            else:
                clusters_under.append(x)

        if params["labeling_strategy"] == "overestimate":
            if clusters_over == []:
                index_cluster_veg = NB_CLUSTERS - 1
            else:
                index_cluster_veg = ratios_surfaces.index(clusters_over[0])
        else:
            if clusters_under == []:
                index_cluster_veg = 0
            else:
                index_cluster_veg = ratios_surfaces.index(clusters_under[-1])

    ratios_surfaces_non_veg = [
        np.where(segments <= i)[0].shape[0] / nb_segments
        for i in range(NB_CLUSTERS)
    ]

    if params["labeling_strategy"] == "nearest":
        takeClosest = lambda num, collection: collection.index(
            min(collection, key=lambda x: abs(x - num))
        )
        index_cluster_no_veg = takeClosest(
            params["pct_non_veg"], ratios_surfaces_non_veg
        )
    else:
        # lists of cluster that overestimate (resp underestimate) the non-vegetation ratio
        clusters_over, clusters_under = [], []
        for x in ratios_surfaces_non_veg:
            if x - params["pct_non_veg"] > 0:
                clusters_over.append(x)
            else:
                clusters_under.append(x)

        if params["labeling_strategy"] == "overestimate":
            if clusters_over == []:
                index_cluster_no_veg = NB_CLUSTERS - 1
            else:
                index_cluster_no_veg = ratios_surfaces_non_veg.index(
                    clusters_over[0]
                )
        else:
            if clusters_under == []:
                index_cluster_no_veg = 0
            else:
                index_cluster_no_veg = ratios_surfaces_non_veg.index(
                    clusters_under[-1]
                )

    nb_clusters_veg = min(index_cluster_veg + 1, NB_CLUSTERS)
    nb_clusters_no_veg = min(index_cluster_no_veg + 1, NB_CLUSTERS)

    logger.debug(
        f"Compute clusters repartition to fit {100*params['pct_veg']}% "
        f"veg and {100*params['pct_non_veg']}% non veg"
    )
    logger.debug(f"{ratios_surfaces=}\n{ratios_surfaces_non_veg=}")
    logger.debug(
        f"{nb_clusters_veg=} ({ratios_surfaces[index_cluster_veg]=}) and "
        f"{nb_clusters_no_veg=} ({ratios_surfaces_non_veg[index_cluster_no_veg]})"
    )

    return nb_clusters_veg, nb_clusters_no_veg


def vegetation_labeling_with_LCM(params: dict, segments):
    """
    Label the segmentation with regards to the clustering step and to an external
    Land Cover Map.
    This methods tries to fix number of vegetation clusters to fit the approximative
    proportion of vegetated areas thanks to the LCM class

    """
    nb_clusters_veg, nb_clusters_non_veg = frac_veg_from_segments(
        segments, params
    )

    nb_clusters_mix = NB_CLUSTERS - nb_clusters_non_veg - nb_clusters_veg

    map_centroid = []
    for i in range(NB_CLUSTERS):
        if i < nb_clusters_non_veg:
            map_centroid.append(NO_VEG_CODE)
        elif i < nb_clusters_non_veg + nb_clusters_mix:
            map_centroid.append(UNDEFINED_VEG)
        else:
            map_centroid.append(VEG_CODE)

    return apply_map(segments, map_centroid)


def frac_low_high_veg_from_segments(
    params: dict, segments_texture, segments_vegetation
):
    # 1. get number of segments with vegetation
    nb_segments_veg = np.where(segments_vegetation >= VEG_CODE)[0].shape[0]

    # 2. for each index of cluster from 8 (NB_CLUSTERS) to 0,
    #    compute ratio of low veg
    ratios_surfaces = [
        np.where(
            segments_texture[np.where(segments_vegetation >= VEG_CODE)] <= i
        )[0].shape[0]
        / nb_segments_veg
        for i in range(NB_CLUSTERS)
    ]

    if params["labeling_strategy"] == "nearest":
        # 3a. select index of the nearest cluster, in term of area covered
        takeClosest = lambda num, collection: collection.index(
            min(collection, key=lambda x: abs(x - num))
        )
        index_cluster = takeClosest(params["pct_low_veg"], ratios_surfaces)
    else:
        # lists of cluster that overestimate (resp underestimate) the non-vegetation ratio
        clusters_over, clusters_under = [], []
        for x in ratios_surfaces:
            if x - params["pct_low_veg"] > 0:
                clusters_over.append(x)
            else:
                clusters_under.append(x)
        if params["labeling_strategy"] == "overestimate":
            # 3b. select index of cluster that overestimate low veg
            if clusters_over == []:
                index_cluster = NB_CLUSTERS - 1
            else:
                index_cluster = ratios_surfaces.index(clusters_over[0])
        else:
            # 3b. select index of cluster that underestimate low veg
            if clusters_under == []:
                index_cluster = 0
            else:
                index_cluster = ratios_surfaces.index(clusters_under[-1])

    nb_clusters_low_veg = min(index_cluster + 1, NB_CLUSTERS)
    nb_clusters_high_veg = NB_CLUSTERS - nb_clusters_low_veg

    logger.debug(
        f"Compute clusters repartition to fit {100*params['pct_low_veg']}% low veg"
    )
    logger.debug(
        f"{ratios_surfaces=} {nb_clusters_low_veg=} {nb_clusters_high_veg=}"
    )

    return nb_clusters_low_veg, nb_clusters_high_veg


def texture_labeling_with_LCM(
    params: dict, segments_texture, segments_vegetation
):

    nb_clusters_low_veg, nb_clusters_high_veg = frac_low_high_veg_from_segments(
        params, segments_texture, segments_vegetation
    )

    map_centroid = []
    for i in range(NB_CLUSTERS):
        if i < nb_clusters_low_veg:
            map_centroid.append(LOW_TEXTURE_CODE)
        else:
            map_centroid.append(HIGH_TEXTURE_CODE)

    textures = np.zeros_like(segments_vegetation)
    textures[np.where(segments_vegetation >= UNDEFINED_VEG)] = apply_map(
        segments_texture[np.where(segments_vegetation >= UNDEFINED_VEG)],
        map_centroid,
    )

    return textures


def vegetation_labeling_with_rule_of_third(params: dict, segments: np.ndarray):
    """
    Label the segmentation with a simple rule of third : first three clusters are NON WATER,
    then the next three are MIX AREA and the last three ones are VEGETATION
    User can adjust this balance by fixing number of supposed vegetation cluster
    """
    index_max_cluster_non_veg = max(
        int((NB_CLUSTERS - params["nb_clusters_veg"]) / 2), 1
    )
    index_max_cluster_mix = max((NB_CLUSTERS - params["nb_clusters_veg"]), 1)
    map_centroid = []

    for i in range(NB_CLUSTERS):
        if i < index_max_cluster_non_veg:
            map_centroid.append(NO_VEG_CODE)
        elif i < index_max_cluster_mix:
            map_centroid.append(UNDEFINED_VEG)
        else:
            map_centroid.append(VEG_CODE)

    return apply_map(segments, map_centroid)


def texture_labeling_with_rule_of_third(
    params: dict, clusters_texture: np.ndarray, clusters_veg: np.ndarray
):
    """
    Label the segmentation with a simple rule of third : first three clusters are NON WATER,
    then the next three are MIX AREA and the last three ones are VEGETATION
    User can adjust this balance by fixing number of supposed vegetation cluster
    """
    # Attribute class by thirds
    index_max_cluster_low_veg = params["nb_clusters_low_veg"]
    map_centroid = []
    for i in range(NB_CLUSTERS):
        if i < index_max_cluster_low_veg:
            map_centroid.append(LOW_TEXTURE_CODE)
        else:
            map_centroid.append(HIGH_TEXTURE_CODE)

    textures = np.zeros_like(clusters_veg)
    textures[np.where(clusters_veg >= UNDEFINED_VEG)] = apply_map(
        clusters_texture[np.where(clusters_veg >= UNDEFINED_VEG)], map_centroid
    )

    return textures


def finalize_task(segments, valid_stack, data):
    """
    Finalize mask : for each pixel in input segmentation,
    return class (low / high vegetation, etc.)

    :param np.ndarray segments: image segments
    :param np.ndarray valid_stack: valid_stack array
    :param np.ndarray data: final cluster data
    :returns: final mask
    """
    
    clustering = data
    # Load Cython module and launch C++ function
    ts_stats = ts.PyStats()

    final_mask = ts_stats.finalize(segments, clustering)

    # Add nodata in final_mask (input_buffers[1] : valid mask)
    final_mask = np.where(valid_stack[0] == 0, final_mask, NODATA_INT8)

    return final_mask


def clean_task(
    im_classif: np.ndarray,
    valid_stack: np.ndarray,
    im_ndvi: np.ndarray,
    remove_small_objects: int,
    remove_small_holes: int,
    binary_dilation: int,
    min_ndvi_veg: float,
) -> np.ndarray:
    """
    Post-processing : remove small holes/objects, apply binary dilation
    on low vegetation and filter using NDVI threshold.

    Parameters
    ----------
    im_classif : np.ndarray
        Segmentation result.
    valid_stack : np.ndarray
        Valid mask.
    im_ndvi : np.ndarray
        NDVI image.
    remove_small_objects : int
    remove_small_holes : int
    binary_dilation : int
    min_ndvi_veg : float

    Returns
    -------
    np.ndarray
        Final processed mask.
    """

    # --- Remove small objects (high vegetation consistency)
    if remove_small_objects:
        high_veg_binary = im_classif > LOW_VEG_CLASS

        high_veg_binary = apply_morpho(
            high_veg_binary.astype(bool),
            "remove_small_holes",
            remove_small_objects,
        ).astype(np.uint8)

        im_classif[
            np.logical_and(
                im_classif == LOW_VEG_CLASS,
                high_veg_binary == 1,
            )
        ] = UNDEFINED_TEXTURE_CLASS

    # --- Low vegetation mask
    low_veg_binary = im_classif == LOW_VEG_CLASS

    # --- Remove small holes
    if remove_small_holes:
        low_veg_binary = apply_morpho(
            low_veg_binary.astype(bool),
            "remove_small_holes",
            remove_small_holes,
        ).astype(np.uint8)

        im_classif[
            np.logical_and(
                im_classif > LOW_VEG_CLASS,
                low_veg_binary == 1,
            )
        ] = LOW_VEG_CLASS

    # --- Binary dilation
    if binary_dilation:
        low_veg_binary = apply_morpho(
            low_veg_binary,
            "binary_dilation",
            binary_dilation,
        ).astype(np.uint8)

        im_classif[
            np.logical_and(
                im_classif > LOW_VEG_CLASS,
                low_veg_binary == 1,
            )
        ] = LOW_VEG_CLASS

    # --- NDVI filtering
    im_classif = np.where(
        im_classif == LOW_VEG_CLASS,
        np.where(im_ndvi > min_ndvi_veg, LOW_VEG_CLASS, 0),
        im_classif,
    )

    im_classif = np.where(
        im_classif > LOW_VEG_CLASS,
        np.where(
            im_ndvi > min_ndvi_veg,
            VEG_CODE + MIDDLE_TEXTURE_CODE,
            0,
        ),
        im_classif,
    )

    # --- Apply nodata mask
    im_classif = np.where(valid_stack == 0, im_classif, NODATA_INT8)

    return im_classif


def segmentation(
    args: argparse.Namespace,
    slurp_manager: slurpContextManager,
    key_ndvi: list,
    key_valid_stack: list,
    ndvi_profile: dict,
):
    """
    Perform SLIC segmentation on NDVI layer using SLURP framework.

    Parameters
    ----------
    args : Namespace
        Runtime configuration and parameters.
    slurp_manager : slurpContextManager
        SLURP execution context.
    key_ndvi : list[str]
        NDVI raster key.
    key_valid_stack : list[str]
        Valid stack raster key.
    ndvi_profile : dict
        NDVI raster profile.
    Returns
    -------
    List[str]
        Segmentation output keys.
    """

    logger.info("Segmentation processing...")

    # ==========================================================
    # INPUTS
    # ==========================================================

    input_keys = [
        key_ndvi[0][0],
        key_valid_stack[0][0],
    ]

    input_profile = deepcopy(ndvi_profile)

    output_profile = eo_utils.single_int32_profile(
        [deepcopy(ndvi_profile)]
    )
    # ==========================================================
    # SEGMENTATION EXECUTION
    # ==========================================================
    future_seg = mp_n_to_m_images_with_mapping(
        inputs=input_keys,
        image_height=input_profile["height"],
        image_width=input_profile["width"],
        output_profiles=[output_profile],
        output_keys=["segmentation_slic"],
        func=segmentation_task,
        func_parameters={
            "slic_seg_size": args.slic_seg_size,
            "slic_compactness": args.slic_compactness,
        },
        context_manager=slurp_manager,
    )

    # ==========================================================
    # OPTIONAL DEBUG SAVE
    # ==========================================================

    #if args.save_mode in ["all", "debug"]:
    output_path = args.vegetationmask.replace(
        ".tif", "_slic.tif"
    )

    slurp_manager.write_tif(
        data=future_seg[0],
        path=output_path,
        target_profile=output_profile,
    )

    return future_seg


def build_stack(args, eoscale_manager):
    """
    Build the required stack of input raster layers for processing.
    """
    # Image VHR
    key_vhr = eoscale_manager.open_raster(raster_path=args.file_vhr)
    args.nodata_vhr = eoscale_manager.get_profile(key_vhr)["nodata"]
    # Valid stack
    key_valid_stack = eoscale_manager.open_raster(raster_path=args.valid_stack)
    # NDXI
    key_ndvi = eoscale_manager.open_raster(raster_path=args.file_ndvi)
    key_ndwi = eoscale_manager.open_raster(raster_path=args.file_ndwi)
    # Texture file
    key_texture = eoscale_manager.open_raster(raster_path=args.file_texture)
    return key_ndvi, key_ndwi, key_vhr, key_texture, key_valid_stack


def postprocess(
    args: argparse.Namespace,
    slurp_manager: slurpContextManager,
    final_seg: list,
    key_valid_stack: list,
    key_ndvi: list,
    output_profile: dict,
):
    """
    Performs morphological closing and post-processing operations
    using SLURP execution framework.

    Parameters
    ----------
    args : Namespace
        Runtime configuration.
    slurp_manager : slurpContextManager
        SLURP context manager.
    final_seg : list[str]
        Segmentation keys.
    key_valid_stack : list[str]
        Valid stack keys.
    key_ndvi : list[str]
        NDVI keys.
    output_profile : dict
        Output raster profile.
    """

    if args.texture_mode == "yes" and (
        args.binary_dilation
        or args.remove_small_objects
        or args.remove_small_holes
    ):

        logger.info("Post-processing segmentation mask")

        # ======================================================
        # COMPUTE STABLE MARGIN
        # ======================================================
        margin = max(
            2 * args.binary_dilation,
            ceil(sqrt(args.remove_small_objects)),
            ceil(sqrt(args.remove_small_holes)),
        )

        input_profile = deepcopy(output_profile)
        output_profile = eo_utils.single_uint8_profile(
            [deepcopy(input_profile)]
        )
        # ======================================================
        # SLURP EXECUTION
        # ======================================================
        final_seg = mp_n_to_m_images(
            inputs=[
                final_seg[0],
                key_valid_stack[0][0],
                key_ndvi[0][0],
            ],
            image_height=input_profile["height"],
            image_width=input_profile["width"],
            output_profiles=[output_profile],
            output_keys=["postprocess"],
            func=clean_task,
            func_parameters=dict(
                remove_small_objects=args.remove_small_objects,
                remove_small_holes=args.remove_small_holes,
                binary_dilation=args.binary_dilation,
                min_ndvi_veg=args.min_ndvi_veg,
            ),
            context_manager=slurp_manager,
            stable_margin=margin,
            binary=True,
        )

    return final_seg


def process_stats(
    args: argparse.Namespace,
    slurp_manager: slurpContextManager,
    future_seg: list,
    key_ndvi: list,
    key_ndwi: list,
    key_texture: list,
    size_result: int,
    mask_valid_indices: np.ndarray,
    input_profile: dict,
):
    """
    Computes per-segment statistics (mean NDVI, NDWI, texture)
    using SLURP multiproc framework.

    Parameters
    ----------
    args : Namespace
        Runtime configuration and file paths.
    slurp_manager : slurpContextManager
        SLURP execution context.
    future_seg : list[str]
        Segmentation output keys.
    key_ndvi : list[str]
        NDVI raster keys.
    key_ndwi : list[str]
        NDWI raster keys.
    key_texture : list[str]
        Texture raster keys.
    size_result : int
        Total number of segments.
    mask_valid_indices : np.ndarray
        Boolean array marking valid segment indices.
    input_profile : dict
        Input raster profile.
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        stats[0] : sum of each primitive (NDVI, NDWI, texture) per segment
        stats[1] : count of pixels per segment
    """

    logger.info("Computing per-segment statistics...")

    params_stats = {"nb_lab": size_result}
    # ======================================================
    # SLURP EXECUTION
    # ======================================================
    stats = mp_n_to_m_scalars(
        inputs=[
            future_seg[0],
            key_ndvi[0][0],
            key_ndwi[0][0],
            key_texture[0][0],
        ],
        image_height=input_profile["height"],
        image_width=input_profile["width"],
        func=compute_stats_image,
        func_parameters=params_stats,
        context_manager=slurp_manager,
        reducer=stats_concatenate,
    )
    # ======================================================
    # COMPUTE MEAN PER SEGMENT
    # ======================================================

    np.seterr(divide="ignore", invalid="ignore")

    # NDVI
    mean_ndvi = stats[0][:size_result]
    mean_ndvi[np.where(mask_valid_indices == 0)] = NODATA_INT16
    mean_ndvi[np.where(mask_valid_indices)] = (
        mean_ndvi[np.where(mask_valid_indices)]
        / stats[1][np.where(mask_valid_indices)]
    )

    # NDWI
    mean_ndwi = stats[0][size_result : 2 * size_result]
    mean_ndwi[np.where(mask_valid_indices == 0)] = NODATA_INT16
    mean_ndwi[np.where(mask_valid_indices)] = (
        mean_ndwi[np.where(mask_valid_indices)]
        / stats[1][np.where(mask_valid_indices)]
    )

    # Texture
    mean_texture = stats[0][2 * size_result : 3 * size_result]
    mean_texture[np.where(mask_valid_indices == 0)] = NODATA_INT16
    mean_texture[np.where(mask_valid_indices)] = (
        mean_texture[np.where(mask_valid_indices)]
        / stats[1][np.where(mask_valid_indices)]
    )

    # ======================================================
    # RETURN
    # ======================================================

    # stats[0] = sum per primitive
    # stats[1] = count per segment
    # downstream clustering uses these arrays
    return stats


def display_infos(
    args,
    end_time,
    t0,
    time_closing,
    time_cluster,
    time_final,
    time_seg,
    time_stack,
    time_stats,
):
    """
    Display information on the time spent on each stage of the processing pipeline.
    """
    logger.info(
        f"**** Vegetation mask for {args.file_vhr} (saved as {args.vegetationmask}) ****"
    )
    logger.info(
        "Total time (user)       :\t" + utils.convert_time(end_time - t0)
    )
    logger.info(
        "- Build_stack           :\t" + utils.convert_time(time_stack - t0)
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
        help=(
            "Minimal mean NDVI to consider a cluster as vegetation "
            "(overloads nb-clusters choice)"
        ),
    )
    group3.add_argument(
        "-max_ndvi_noveg",
        type=int,
        help=(
            "Maximal mean NDVI to consider a cluster as non-vegetation "
            "(overloads nb-clusters choice)"
        ),
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
        help=(
            "Maximal texture value to consider a cluster as low vegetation "
            "(overloads nb-clusters choice)"
        ),
    )
    group3.add_argument(
        "-autolabel",
        action="store_true",
        help="Automatic labeling method that will fit supposed ratios of vegetation "
        "(non-vegetation) areas (as observed in a global LCM)",
    )
    group3.add_argument(
        "-labeling_strategy",
        choices=["nearest", "overestimate", "underestimate"],
        dest="labeling_strategy",
        default="nearest",
        help="In case of automatic labeling, choose the cluster that gives the nearest ratio,"
        " or that overestimage a little bit vegetation(resp underestimate)",
    )
    group3.add_argument(
        "-pct_veg",
        type=float,
        help="Pourcentage of vegetation pixels in the global land cover map",
    )
    group3.add_argument(
        "-pct_low_veg",
        type=float,
        help="Pourcentage of low vegetation pixels in the global land cover map",
    )
    group3.add_argument(
        "-pct_high_veg",
        type=float,
        help="Pourcentage of high vegetation pixels in the global land cover map",
    )
    group3.add_argument(
        "-pct_non_veg",
        type=float,
        help="Pourcentage of non vegetation pixels in the global land cover map",
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

    with open("args_list.json", "w") as f:
        json.dump(arglist, f)

    return vars(args)


def slurp_vegetationmask(
    main_config: str,
    debug: bool,
    logs_to_file: bool,
    user_config: str,
    file_vhr: str,
    valid_stack: bool,
    file_ndvi: str,
    file_ndwi: str,
    file_texture: str,
    texture_mode: str,
    filter_texture: int,
    save_mode: str,
    slic_seg_size: int,
    slic_compactness: float,
    nb_clusters_veg: int,
    min_ndvi_veg: int,
    max_ndvi_noveg: int,
    non_veg_clusters: bool,
    nb_clusters_low_veg: int,
    autolabel: bool,
    labeling_strategy: str,
    pct_veg: float,
    pct_low_veg: float,
    pct_high_veg: float,
    pct_non_veg: float,
    max_texture_th: int,
    binary_dilation: int,
    remove_small_objects: int,
    remove_small_holes: int,
    vegetationmask: str,
    n_workers: int,
    tile_max_size: int,
    multiproc_context: str,
):
    """
    Main API to compute vegetation mask using slurpContextManager.
    """

    # =====================================================
    # LOAD CONFIG
    # =====================================================

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
        if locals().get(param) is not None:
            argsdict[param] = locals()[param]

    logger.info("--" * 50)
    logger.info("SLURP - Vegetation mask\n")
    logger.info(f"JSON data loaded: {main_config}")

    args = argparse.Namespace(**argsdict)

    if args.debug:
        logger.handlers[0].setLevel(logging.DEBUG)

    logger.debug(f"{argsdict=}")

    # =====================================================
    # SLURP CONTEXT
    # =====================================================

    params = {
        "nb_max_workers": args.n_workers,
        "developer_mode": args.debug,
        "method": "mem",
        "mp_context": args.multiproc_context,
        "output_dir": path.dirname(args.file_vhr),
    }

    with slurpContextManager(params, tile_mode=True, tile_max_size=args.tile_max_size) as slurp_manager:

        try:

            t0 = time.time()

            # =====================================================
            # BUILD STACK
            # =====================================================

            logger.info("[0] Step: Build stack")

            (
                ndvi,
                ndwi,
                vhr,
                texture,
                valid_stack,
                vhr_profile,
                valid_stack_profile,
                ndvi_profile
            ) = build_stack_vegetation(args, slurp_manager)

            time_stack = time.time()

            # =====================================================
            # SEGMENTATION
            # =====================================================

            logger.info("[1] Step: Segmentation")

            segments = segmentation(
                args,
                slurp_manager,
                ndvi,
                valid_stack,
                ndvi_profile
            )

            time_seg = time.time()

            # =====================================================
            # COMPUTE VALID SEGMENTS
            # =====================================================

            logger.info("[2] Step: Segment validity")
            res_seg = segments[0]
            size_result = np.max(res_seg) + 1

            ts_stats = ts.PyStats()

            mask_valid_indices = ts_stats.compute_mask_valid_indices(
                res_seg,
                size_result,
            )

            # =====================================================
            # STATS
            # =====================================================

            logger.info("[3] Step: Compute statistics")

            stats = process_stats(
                args,
                slurp_manager,
                segments,
                ndvi,
                ndwi,
                texture,
                size_result,
                mask_valid_indices,
                ndvi_profile
            )

            time_stats = time.time()

            # =====================================================
            # CLUSTERING
            # =====================================================

            logger.info("[4] Step: Clustering")

            pred_veg, sorted_ndvi_centroids = clustering_vegetation(
                vars(args),
                size_result,
                stats[0],
                mask_valid_indices,
            )

            if args.autolabel:
                clusters_veg = vegetation_labeling_with_LCM(
                    vars(args), pred_veg
                )
            else:
                clusters_veg = vegetation_labeling_with_rule_of_third(
                    vars(args), pred_veg
                )

            pred_texture, _ = clustering_texture(
                vars(args),
                size_result,
                stats[0],
                clusters_veg,
                mask_valid_indices,
            )
            if args.autolabel:
                clusters_low_high_veg = texture_labeling_with_LCM(
                    vars(args), pred_texture, clusters_veg
                )
            else:
                clusters_low_high_veg = texture_labeling_with_rule_of_third(
                    vars(args), pred_texture, clusters_veg
                )

            clusters = clusters_veg + clusters_low_high_veg
            time_cluster = time.time()

            # =====================================================
            # FINALIZE MASK
            # =====================================================

            logger.info("[5] Step: Finalize mask")

            # final tab
            final_clusters = np.zeros(size_result)
            final_clusters[np.where(mask_valid_indices)] = clusters
            final_clusters[np.where(mask_valid_indices == 0)] = (
                0  # TODO : -1 or 0 ? it will be masked by valid_stack at the end
            )

            final_mask = mp_n_to_m_images(
                inputs=[segments[0], valid_stack[0][0]],
                image_height=vhr_profile["height"],
                image_width=vhr_profile["width"],
                output_profiles=[
                    eo_utils.single_uint8_profile([vhr_profile])
                ],
                output_keys=[path.basename(args.vegetationmask)],
                func=finalize_task,
                func_parameters={"data": final_clusters},
                context_manager=slurp_manager,
                binary=True,
            )
            if args.save_mode == "debug":
                # Save intermediate masks
                output_path = args.vegetationmask.replace(
                    ".tif", "_before_clean.tif"
                )
                slurp_manager.write_tif(
                    data=final_mask[0],
                    path=output_path,
                    target_profile=eo_utils.single_uint8_profile(
                        [vhr_profile]
                    ),
                )

            time_final = time.time()

            # =====================================================
            # POSTPROCESS
            # =====================================================

            logger.info("[6] Step: Postprocess")

            vars(args)["min_ndvi_veg"] = sorted_ndvi_centroids[
                -args.nb_clusters_veg
            ]

            final_mask = postprocess(
                args,
                slurp_manager,
                final_mask,
                valid_stack,
                ndvi,
                eo_utils.single_uint8_profile([vhr_profile])
            )

            time_post = time.time()
            # =====================================================
            # WRITE OUTPUT
            # =====================================================

            slurp_manager.write_tif(
                data=final_mask[0],
                path=args.vegetationmask,
                target_profile=eo_utils.single_uint8_profile(
                    [vhr_profile]
                ),
            )

            t1 = time.time()

            logger.info(
                "Total time (user):\t" + utils.convert_time(t1 - t0)
            )

        except Exception:
            logger.error("Unexpected error:", exc_info=True)
            traceback.print_exc()

    logger.info("End of vegetationmask step\n")


def main():
    """
    Main function to run the vegetation mask computation.
    It parses the command line arguments and calls the slurp_vegetationmask function.
    """
    args = getarguments()
    slurp_vegetationmask(**args)


if __name__ == "__main__":
    main()
