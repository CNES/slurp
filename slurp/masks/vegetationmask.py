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
from copy import deepcopy
from math import ceil, sqrt
from os import path

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.signal import savgol_filter
from skimage.segmentation import slic
from sklearn.cluster import KMeans

try:
    import maxflow
except ImportError:  # optional dependency, only needed by the graph cut step
    maxflow = None

# Cython module to compute stats
import stats as ts
from slurp import __version__
from slurp.eomultiprocessing.slurp_executor import (
    mp_n_to_m_images,
    mp_n_to_m_images_with_mapping,
    mp_n_to_m_scalars,
)
from slurp.eomultiprocessing.slurp_manager import slurpContextManager
from slurp.eomultiprocessing.utils import read, read_and_get_profile
from slurp.post_process.morphology import apply_morpho
from slurp.tools import profile_utils as eo_utils
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
    return np.asarray(map_centroids)[np.asarray(pred, dtype=np.intp)]


def rank_of_centroids(centroids: np.ndarray) -> np.ndarray:
    """
    Rank of each cluster once sorted by increasing centroid value.

    Replaces ``[sorted_values.index(v) for v in values]`` which silently
    returned the same rank twice when two centroids were exactly equal (one
    cluster then became unreachable in the label mapping).
    """
    centroids = np.asarray(centroids).ravel()
    order = np.argsort(centroids, kind="stable")
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(order.size)
    return ranks


def as_bool_mask(mask) -> np.ndarray:
    """
    Boolean view of a 0/1 mask.
    """
    mask = np.asarray(mask)
    if mask.dtype == bool:
        return mask
    return mask.astype(bool)


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
    key_shadowmask : list
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

    key_texture, profile_texture = read_and_get_profile(args.file_texture)

    # ======================================================
    # SHADOW MASK
    # ======================================================

    key_shadowmask = read(args.shadowmask)

    # ======================================================
    # RETURN (SLURP FORMAT)
    # ======================================================

    return (
        [key_ndvi],
        [key_ndwi],
        [key_vhr],
        [key_texture],
        [key_valid_stack],
        [key_shadowmask],
        profile_vhr,
        profile_texture,
        profile_ndvi,
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

    # nseg = int(ndvi.shape[2] * ndvi.shape[1] / params["slic_seg_size"])
    # nseg cannot be equal to 0, because calling function already checked
    # there were valid pixels to segment...
    nb_valid = int(np.count_nonzero(ndvi != NODATA_INT16))
    nseg = nb_valid // int(params["slic_seg_size"])
    if nseg < 1:
        logger.debug(f"Segments number : 0 !! Divide by zero {ndvi.shape=}")
        nseg = 1

    # Note : we read NDVI image.
    # Estimation of the max number of segments (ie : each segment is > 100 pixels)
    res_seg = slic(
        ndvi.astype("double"),
        compactness=float(params["slic_compactness"]),
        n_segments=nseg,
        sigma=1,
        channel_axis=None,
        start_label=1,
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


# Stats #
def compute_stats_image(
    segments: np.ndarray,
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    texture: np.ndarray,
    nb_lab: int,
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
    Concatenate statistics coming from multiple sub-tiles.

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

    valid = as_bool_mask(mask_valid_indices)

    ndvi = stats[0:size_result]
    ndwi = stats[size_result : 2 * size_result]
    logger.debug(f"Before NODATA removal {ndvi.shape=}")
    ndvi = ndvi[valid]
    ndwi = ndwi[valid]
    vec_predic = np.stack((ndvi, ndwi), axis=1).astype(np.float32, copy=False)
    logger.debug(
        f"{int(valid.sum())=} -> after NODATA removal {ndvi.shape=}"
    )

    pred_veg = kmeans_rad_indices.fit_predict(vec_predic)

    ndvi_values = kmeans_rad_indices.cluster_centers_[:, 0]
    sorted_ndvi = np.sort(ndvi_values).tolist()

    sorted_clusters = rank_of_centroids(ndvi_values)
    logger.debug(
        f"1st clustering : NDVI centroids : {sorted_ndvi} {sorted_clusters=}"
    )
    pred_veg_sorted = sorted_clusters[pred_veg]

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
    valid = as_bool_mask(mask_valid_indices)
    veg_values = mean_texture[valid]

    is_veg = clustering >= UNDEFINED_VEG
    texture_values = veg_values[is_veg]

    threshold_max = np.percentile(texture_values, params["filter_texture"])
    data_textures = np.minimum(texture_values, threshold_max).astype(
        np.float32, copy=False
    )

    kmeans_texture = KMeans(
        n_clusters=NB_CLUSTERS,
        init="k-means++",
        n_init=5,
        verbose=0,
        random_state=712,
    )
    pred_texture = kmeans_texture.fit_predict(data_textures.reshape(-1, 1))

    centroids_texture = kmeans_texture.cluster_centers_[:, 0]
    sorted_texture = np.sort(centroids_texture).tolist()

    sorted_clusters = rank_of_centroids(centroids_texture)
    logger.debug(f"2nd clustering : Texture centroids : {sorted_texture}")
    textures = np.zeros(clustering.shape[0], dtype=np.uint8)
    textures[is_veg] = sorted_clusters[pred_texture]
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
        np.count_nonzero(segments >= i) / nb_segments
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
        np.count_nonzero(segments <= i) / nb_segments
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
    # data crosses the process boundary once per tile : it is shipped as uint8
    # (the nomenclature never exceeds 23) and restored to the dtype expected by
    # the Cython kernel here. On a large scene that table weighs tens of
    # megabytes and was pickled in float64 for every single tile.
    clustering = np.ascontiguousarray(data, dtype=np.float64)

    # Load Cython module and launch C++ function
    ts_stats = ts.PyStats()

    final_mask = ts_stats.finalize(segments, clustering)

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
    apply_ndvi_filter: bool = True,
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
    apply_ndvi_filter : bool
        Re-apply the per-pixel NDVI threshold at the end. Must be disabled when
        the graph cut ran before : the refinement decides in shadows and in low
        contrast areas precisely because the raw NDVI is not trustworthy there,
        and thresholding again undoes it pixel per pixel.

    Returns
    -------
    np.ndarray
        Final processed mask.
    """

    output_shape = np.asarray(im_classif).shape
    im_classif = np.squeeze(np.asarray(im_classif)).copy()
    im_ndvi = np.squeeze(np.asarray(im_ndvi))
    valid = np.squeeze(np.asarray(valid_stack)) == 0

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

    if apply_ndvi_filter:
        im_classif = np.where(
            im_classif == LOW_VEG_CLASS,
            np.where(
                im_ndvi > min_ndvi_veg,
                LOW_VEG_CLASS,
                UNDEFINED_VEG + LOW_TEXTURE_CODE,
            ),
            im_classif,
        )

        im_classif = np.where(
            im_classif > LOW_VEG_CLASS,
            np.where(
                im_ndvi > min_ndvi_veg,
                VEG_CODE + MIDDLE_TEXTURE_CODE,
                UNDEFINED_VEG + MIDDLE_TEXTURE_CODE,
            ),
            im_classif,
        )

    # --- Apply nodata mask
    im_classif = np.where(valid, im_classif, NODATA_INT8).astype(np.uint8)

    return im_classif.reshape(output_shape)


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

    output_profile = eo_utils.single_int32_profile([deepcopy(ndvi_profile)])
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

    if args.save_mode in ["all", "debug"]:
        output_path = args.vegetationmask.replace(".tif", "_slic.tif")

        slurp_manager.write_tif(
            data=future_seg[0],
            path=output_path,
            target_profile=output_profile,
        )

    return future_seg


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
            func_parameters={
                "remove_small_objects": args.remove_small_objects,
                "remove_small_holes": args.remove_small_holes,
                "binary_dilation": args.binary_dilation,
                "min_ndvi_veg": args.min_ndvi_veg,
                # the graph cut already arbitrated the ambiguous pixels
                "apply_ndvi_filter": not bool(
                    getattr(args, "graphcut", False)
                ),
            },
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

    valid = as_bool_mask(mask_valid_indices)
    counts = stats[1][valid]

    for offset, name in enumerate(("NDVI", "NDWI", "texture")):
        mean = stats[0][offset * size_result : (offset + 1) * size_result]
        mean[~valid] = NODATA_INT16
        mean[valid] = mean[valid] / counts
        logger.debug(f"{name} means computed on {counts.size} segments")

    # ======================================================
    # RETURN
    # ======================================================

    # stats[0] = sum per primitive
    # stats[1] = count per segment
    # downstream clustering uses these arrays
    return stats


# GRAPH CUT REFINEMENT (ALPHA-EXPANSION)

# Max-flow only solves the two-label case exactly. For K labels we use
# alpha-expansion (Boykov, Veksler & Zabih, 2001) : for each label alpha we
# solve the binary problem "every pixel either keeps its current label or takes
# alpha".

GRAPHCUT_LABELS = np.array(
    [
        NO_VEG_CODE,
        UNDEFINED_VEG + LOW_TEXTURE_CODE,
        UNDEFINED_VEG + MIDDLE_TEXTURE_CODE,
        UNDEFINED_VEG + HIGH_TEXTURE_CODE,
        VEG_CODE + LOW_TEXTURE_CODE,
        VEG_CODE + MIDDLE_TEXTURE_CODE,
        VEG_CODE + HIGH_TEXTURE_CODE,
    ],
    dtype=np.int32,
)

GRAPHCUT_LABEL_NAMES = {
    NO_VEG_CODE: "non vegetation",
    UNDEFINED_VEG + LOW_TEXTURE_CODE: "undefined veg / low texture",
    UNDEFINED_VEG + MIDDLE_TEXTURE_CODE: "undefined veg / middle texture",
    UNDEFINED_VEG + HIGH_TEXTURE_CODE: "undefined veg / high texture",
    VEG_CODE + LOW_TEXTURE_CODE: "low vegetation",
    VEG_CODE + MIDDLE_TEXTURE_CODE: "vegetation / middle texture",
    VEG_CODE + HIGH_TEXTURE_CODE: "high vegetation",
}

GRAPHCUT_INDEX = {int(code): i for i, code in enumerate(GRAPHCUT_LABELS)}
GRAPHCUT_NB_LABELS = len(GRAPHCUT_LABELS)

CODE_TO_INDEX = np.full(256, -1, dtype=np.int32)
for _code, _index in GRAPHCUT_INDEX.items():
    CODE_TO_INDEX[_code] = _index

VEGETATION_AXIS = {NO_VEG_CODE: 0.0, UNDEFINED_VEG: 0.45, VEG_CODE: 1.0}

TEXTURE_AXIS = {
    LOW_TEXTURE_CODE: 0.0,
    MIDDLE_TEXTURE_CODE: 0.5,
    HIGH_TEXTURE_CODE: 1.0,
}

GRAPHCUT_CONFIDENCE = {
    NO_VEG_CODE: 0.70,
    UNDEFINED_VEG + LOW_TEXTURE_CODE: 0.35,
    UNDEFINED_VEG + MIDDLE_TEXTURE_CODE: 0.30,
    UNDEFINED_VEG + HIGH_TEXTURE_CODE: 0.35,
    VEG_CODE + LOW_TEXTURE_CODE: 0.60,
    VEG_CODE + MIDDLE_TEXTURE_CODE: 0.50,
    VEG_CODE + HIGH_TEXTURE_CODE: 0.60,
}

EPS = 1e-12

# Default values for every graph cut parameter. They are injected into argsdict
# when the JSON configuration does not define them, so that existing
# configuration files keep working unchanged.
GRAPHCUT_DEFAULTS = {
    "graphcut": False,
    "graphcut_lambda": 3.0,
    "graphcut_truncation": 0.75,
    "graphcut_tau": 0.35,
    "graphcut_texture_spread": 0.15,
    "graphcut_ndvi_scale": 1000.0,
    "graphcut_ndvi_threshold": None,
    "graphcut_ndvi_slope": 0.10,
    "graphcut_ndvi_weight": 0.5,
    "graphcut_shadow_factor": 0.25,
    "graphcut_attribute_weights": (1.0, 0.3, 0.5),
    "graphcut_contrast_sigma": 1.0,
    "graphcut_contrast_kernel": "gauss",
    "graphcut_diffusion_iterations": 0,
    "graphcut_lambda_absolute": None,
    "graphcut_scale_factor": 2,
    "graphcut_refine_band": 0,
    "graphcut_cycles": 6,
    "graphcut_margin": 64,
    "graphcut_fill_holes": True,
    "graphcut_hole_max_area": 200,
    "graphcut_hole_max_area_shadow": 2000,
    "graphcut_visible_bands": (0, 1, 2),
}


def set_graphcut_defaults(argsdict: dict) -> dict:
    """
    Fill in the graph cut parameters that the configuration does not define.

    :param dict argsdict: dictionary of arguments, updated in place
    :returns: the same dictionary
    """
    for key, value in GRAPHCUT_DEFAULTS.items():
        if argsdict.get(key) is None:
            argsdict[key] = value
    return argsdict


def build_label_positions(texture_spread: float) -> np.ndarray:
    """
    Embed every label of the nomenclature in a small 2D space.

    Axis 1 is the vegetation axis and carries the semantics, axis 2 is the
    texture axis. Building the transition matrix from a Euclidean embedding
    guarantees the triangular inequality by construction, hence the validity of
    alpha-expansion.

    :param float texture_spread: extent of the texture axis relative to the
        vegetation axis
    :returns: array of shape (nb_labels, 2)
    """
    positions = np.zeros((GRAPHCUT_NB_LABELS, 2), dtype=float)

    for code, index in GRAPHCUT_INDEX.items():
        if code == NO_VEG_CODE:
            positions[index] = (VEGETATION_AXIS[NO_VEG_CODE], 0.0)
            continue
        vegetation_code = (code // 10) * 10
        texture_code = code - vegetation_code
        positions[index] = (
            VEGETATION_AXIS[vegetation_code],
            texture_spread * TEXTURE_AXIS[texture_code],
        )

    return positions

def transition_matrix(positions: np.ndarray, truncation: float) -> np.ndarray:
    """
    Compute V[a, b] : cost of a transition between two neighbouring labels.
    """
    nb_labels = positions.shape[0]
    distances = np.zeros((nb_labels, nb_labels))

    for a in range(nb_labels):
        for b in range(nb_labels):
            delta = positions[a] - positions[b]
            distances[a, b] = np.sqrt((delta ** 2).sum())

    distances /= distances.max()
    return np.minimum(distances, truncation)


def check_metric(transitions: np.ndarray, tol: float = 1e-9) -> dict:
    """
    Check that the transition matrix is a metric.

    Alpha-expansion requires V(a, a) = 0, V symmetric and the triangular
    inequality V(a, b) <= V(a, c) + V(c, b). If any of these fails, the binary
    sub-problems become non submodular and max-flow no longer applies.

    :param np.ndarray transitions: transition matrix
    :param float tol: numerical tolerance
    :returns: dict of boolean checks
    """
    return {
        "null_diagonal": bool(np.allclose(np.diag(transitions), 0, atol=tol)),
        "symmetric": bool(np.allclose(transitions, transitions.T, atol=tol)),
        "triangular_inequality": bool(
            (
                transitions[:, :, None]
                <= transitions[:, None, :] + transitions.T[None, :, :] + tol
            ).all()
        ),
    }


def prior_confusion_matrix(
    positions: np.ndarray, confidence: dict, tau: float
) -> np.ndarray:
    """
    Build M[j, k] = P(true class = k | observed class = j).

    The diagonal is the confidence granted to the observed class, the remaining
    mass is spread over the other classes according to the semantic distance :
    a high vegetation pixel is far more likely to actually be low vegetation
    than bare soil.

    :param np.ndarray positions: label embedding
    :param dict confidence: confidence per label code
    :param float tau: spread of the residual probability over the other classes
        (small value = confusions only between neighbouring classes)
    :returns: np.ndarray of shape (nb_labels, nb_labels)
    """
    distances = np.sqrt(
        ((positions[:, None, :] - positions[None, :, :]) ** 2).sum(-1)
    )
    distances /= distances.max()

    weights = np.exp(-distances / tau)
    np.fill_diagonal(weights, 0.0)
    weights /= weights.sum(axis=1, keepdims=True)

    diagonal = np.array([confidence[int(code)] for code in GRAPHCUT_LABELS])[
        :, None
    ]

    confusion = weights * (1.0 - diagonal)
    confusion[np.arange(GRAPHCUT_NB_LABELS), np.arange(GRAPHCUT_NB_LABELS)] = (
        diagonal.ravel()
    )

    return confusion


# --- Data term --------------------------------------------------------------


def observed_labels(im_classif: np.ndarray) -> tuple:
    """
    Convert the mask codes into label indices.

    :param np.ndarray im_classif: mask holding nomenclature codes
    :returns: (indices array, boolean array of pixels inside the nomenclature)
    """
    observed = CODE_TO_INDEX[np.asarray(im_classif).astype(np.uint8)]
    known = observed >= 0

    # Anything outside the nomenclature is handled as the most uncertain class
    observed = np.where(
        known, observed, GRAPHCUT_INDEX[UNDEFINED_VEG + MIDDLE_TEXTURE_CODE]
    ).astype(np.int32)

    return observed, known


def unary_cost(observed: np.ndarray, confusion: np.ndarray) -> np.ndarray:
    """
    Data term D_p(k) = -log M[observed_p, k].

    The logarithm is taken on the 7x7 matrix, not on the (H, W, nb_labels)
    array : both give the same result but the second one evaluates several
    million logarithms per tile.

    :param np.ndarray observed: label indices of the input mask
    :param np.ndarray confusion: prior confusion matrix
    :returns: np.ndarray of shape (H, W, nb_labels), float32
    """
    log_confusion = -np.log(np.clip(confusion, EPS, 1.0)).astype(np.float32)

    return log_confusion[observed]


def add_ndvi_evidence(
    unary: np.ndarray,
    ndvi: np.ndarray,
    positions: np.ndarray,
    threshold: float,
    slope: float,
    weight: float,
) -> np.ndarray:
    """
    Add a spectral evidence that does not depend on the SLIC mask.

    A sigmoid on the NDVI gives P(pixel is vegetated). Each label is charged
    according to its own position along the vegetation axis : a label at 1.0
    pays when the NDVI is low, a label at 0.0 pays when it is high, and the
    UNDEFINED_VEG labels sit at 0.45 so they stay nearly neutral.

    :param np.ndarray unary: data term, shape (H, W, nb_labels)
    :param np.ndarray ndvi: NDVI in physical units
    :param np.ndarray positions: label embedding
    :param float threshold: NDVI value at the centre of the sigmoid
    :param float slope: width of the sigmoid transition
    :param float weight: relative weight of this evidence
    :returns: updated data term
    """
    if weight <= 0:
        return unary

    vegetated = 1.0 / (
        1.0
        + np.exp(
            -(ndvi.astype(np.float32, copy=False) - np.float32(threshold))
            / np.float32(slope)
        )
    )

    for label, value in enumerate(positions[:, 0].astype(np.float32)):
        probability = value * vegetated + (1.0 - value) * (1.0 - vegetated)
        cost = -np.log(np.clip(probability, EPS, 1.0)).astype(np.float32)
        unary[:, :, label] += cost * np.float32(weight)

    return unary


def weaken_unaries(
    unary: np.ndarray, mask: np.ndarray, factor: float
) -> np.ndarray:
    """
    Flatten the data term where no reliable information is available.

    In shadows the radiometry carries no usable evidence, so the data term is
    crushed and the label is decided by the neighbourhood instead.

    :param np.ndarray unary: data term
    :param np.ndarray mask: boolean mask of unreliable pixels
    :param float factor: multiplicative factor applied to the data term
    :returns: updated data term
    """
    if not mask.any():
        return unary

    unary[mask] *= factor

    return unary


# --- Regularisation term ----------------------------------------------------


def neighbour_differences(field: np.ndarray, axis: int) -> np.ndarray:
    """
    Difference between 4-connected neighbours along one axis.

    :param np.ndarray field: 2D array
    :param int axis: 1 for horizontal pairs, 0 for vertical pairs
    :returns: np.ndarray of shape (H, W-1) or (H-1, W)
    """
    if axis == 1:
        return field[:, :-1] - field[:, 1:]

    return field[:-1, :] - field[1:, :]


def contrast_weights(
    attributes: list,
    weights: list,
    axis: int,
    sigma: float,
    kernel: str = "gauss",
) -> np.ndarray:
    """
    Local contrast between neighbours, in a multi-attribute space.

    The contrast does not rely on NDVI alone : a low/high vegetation boundary
    is often invisible in NDVI but sharp in texture and luminance. Attributes
    are stacked, each with its own weight, and the distance between neighbours
    is computed in that space.

    Two kernels are available :
      * "gauss"  : exp(-d^2 / 2 sigma^2), decreases fast, very much all or
        nothing
      * "cauchy" : 1 / (1 + (d / sigma)^2), heavier tail, keeps some
        regularisation across medium contrasts, which avoids freezing the
        errors of the input mask

    :param list attributes: list of normalized 2D arrays
    :param list weights: relative importance of each attribute
    :param int axis: 1 for horizontal pairs, 0 for vertical pairs
    :param float sigma: contrast scale, in normalized attribute units
    :param str kernel: "gauss" or "cauchy"
    :returns: np.ndarray of edge weights
    """
    squared = None
    for attribute, weight in zip(attributes, weights):
        difference = neighbour_differences(attribute, axis)
        difference *= difference
        difference *= np.float32(weight)
        if squared is None:
            squared = difference
        else:
            squared += difference

    if kernel == "gauss":
        return np.exp(-squared / np.float32(2.0 * sigma * sigma)).astype(
            np.float32
        )

    if kernel == "cauchy":
        return (1.0 / (1.0 + squared / np.float32(sigma * sigma))).astype(
            np.float32
        )

    raise ValueError(
        f"Unknown contrast kernel {kernel!r}, expected 'gauss' or 'cauchy'"
    )


def normalize_with(
    field: np.ndarray, centre: float, scale: float
) -> np.ndarray:
    """
    Centre and reduce an attribute with statistics computed over the whole
    image, so that two adjacent tiles share the same normalization.

    :param np.ndarray field: attribute to normalize
    :param float centre: global centre
    :param float scale: global scale
    :returns: normalized attribute
    """
    return ((field - centre) / (scale + EPS)).astype(np.float32)


def automatic_kappa(gradient_magnitude: np.ndarray, bins: int = 300) -> float:
    """
    Estimate the diffusion threshold from the gradient histogram.

    The threshold is taken at the first inflexion point of the gradient
    magnitude distribution located after its mode. The estimation may fail on
    degenerate histograms, in which case the caller falls back on a percentile.

    :param np.ndarray gradient_magnitude: gradient magnitude
    :param int bins: number of histogram bins
    :returns: float threshold
    """
    counts, edges = np.histogram(
        gradient_magnitude.ravel(), bins=bins, density=True
    )
    centers = (edges[:-1] + edges[1:]) / 2

    smoothed = savgol_filter(counts, window_length=11, polyorder=3)
    phi = centers * smoothed
    second_derivative = np.gradient(np.gradient(phi, centers), centers)

    interpolated = interp1d(centers, second_derivative, kind="cubic")
    index_max_phi = np.argmax(phi)

    sign_changes = np.where(np.diff(np.sign(second_derivative)))[0]
    sign_changes = sign_changes[sign_changes > index_max_phi]

    if len(sign_changes) == 0:
        raise ValueError("no inflexion point found after the histogram mode")

    lower = centers[sign_changes[0]]
    upper = centers[sign_changes[0] + 1]

    return brentq(interpolated, lower, upper)


def anisotropic_diffusion(
    index: np.ndarray,
    gamma: float = 0.1,
    max_iter: int = 25,
    sigma: float = 0.3,
    kappa: float = None,
) -> np.ndarray:
    """
    Perona-Malik anisotropic diffusion with an automatic threshold.

    Smooths the inside of homogeneous areas while preserving the edges, which
    makes the contrast term less noisy. Falls back on a percentile threshold if
    the automatic estimation fails.

    :param np.ndarray index: 2D array to smooth
    :param float gamma: diffusion step
    :param int max_iter: number of iterations. Note that the threshold is
        estimated on the histogram of the tile, so enabling the diffusion makes
        the contrast term slightly tile dependent
    :param float sigma: smoothing applied before computing the gradient
    :returns: smoothed array
    """
    if max_iter <= 0:
        return index.astype(np.float32)

    image = np.array(index, dtype=np.float32, copy=True)

    if kappa is None:
        gradient_y, gradient_x = np.gradient(image)
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        try:
            kappa = automatic_kappa(magnitude)
        except Exception:
            kappa = float(np.percentile(magnitude, 90)) + EPS
            logger.debug(
                "Automatic kappa estimation failed, "
                f"falling back on the 90th percentile ({kappa=})"
            )

        if not np.isfinite(kappa) or kappa <= 0:
            kappa = float(np.percentile(magnitude, 90)) + EPS

    inverse_kappa2 = np.float32(1.0 / (kappa * kappa))
    gamma = np.float32(gamma)
    laplacian = np.empty_like(image)

    for _ in range(max_iter):
        smoothed = ndi.gaussian_filter(image, sigma=sigma, mode="reflect")
        gradient_y, gradient_x = np.gradient(smoothed)

        np.square(gradient_x, out=gradient_x)
        np.square(gradient_y, out=gradient_y)
        gradient_x += gradient_y
        gradient_x *= -inverse_kappa2
        conductance = np.exp(gradient_x, out=gradient_x)

        np.multiply(image, np.float32(-4.0), out=laplacian)
        laplacian[1:, :] += image[:-1, :]
        laplacian[0, :] += image[0, :]
        laplacian[:-1, :] += image[1:, :]
        laplacian[-1, :] += image[-1, :]
        laplacian[:, 1:] += image[:, :-1]
        laplacian[:, 0] += image[:, 0]
        laplacian[:, :-1] += image[:, 1:]
        laplacian[:, -1] += image[:, -1]

        laplacian *= conductance
        laplacian *= gamma
        image += laplacian

    return image


# --- Alpha-expansion --------------------------------------------------------


def prepare_edges(weights_h: np.ndarray, weights_v: np.ndarray) -> tuple:
    """
    Pre-compute the flattened float64 edge weights and the neighbour slices.

    PyMaxflow works in double precision, so the weights have to be converted
    before every graph construction. Doing it here means one conversion per
    optimisation instead of one per alpha and per cycle (7 labels x 6 cycles =
    42 full copies of both weight maps for a single tile).

    :returns: tuple of (flat weights, slice of p, slice of q)
    """
    return (
        (
            np.ascontiguousarray(weights_h, dtype=np.float64).ravel(),
            (slice(None), slice(0, -1)),
            (slice(None), slice(1, None)),
        ),
        (
            np.ascontiguousarray(weights_v, dtype=np.float64).ravel(),
            (slice(0, -1), slice(None)),
            (slice(1, None), slice(None)),
        ),
    )


def boundary_band(
    labels: np.ndarray, radius: int, valid: np.ndarray = None
) -> np.ndarray:
    """
    Boolean mask of the pixels within ``radius`` of a label boundary.

    :param np.ndarray labels: label indices
    :param int radius: dilation radius, in pixels
    :param np.ndarray valid: optional validity mask
    :returns: boolean mask
    """
    boundary = np.zeros(labels.shape, dtype=bool)

    difference = labels[:, :-1] != labels[:, 1:]
    boundary[:, :-1] |= difference
    boundary[:, 1:] |= difference

    difference = labels[:-1, :] != labels[1:, :]
    boundary[:-1, :] |= difference
    boundary[1:, :] |= difference

    if radius > 0:
        boundary = ndi.binary_dilation(
            boundary,
            ndi.generate_binary_structure(2, 2),
            iterations=int(radius),
        )

    if valid is not None:
        boundary &= valid

    return boundary


def graphcut_energy(
    labels: np.ndarray,
    unary: np.ndarray,
    transitions: np.ndarray,
    weights_h: np.ndarray,
    weights_v: np.ndarray,
) -> float:
    """
    Total energy of a labelling.

    :param np.ndarray labels: label indices, shape (H, W)
    :param np.ndarray unary: data term, shape (H, W, nb_labels)
    :param np.ndarray transitions: transition matrix
    :param np.ndarray weights_h: horizontal edge weights, shape (H, W-1)
    :param np.ndarray weights_v: vertical edge weights, shape (H-1, W)
    :returns: float energy
    """
    energy = float(np.take_along_axis(unary, labels[:, :, None], axis=2).sum())
    energy += float(
        (weights_h * transitions[labels[:, :-1], labels[:, 1:]]).sum()
    )
    energy += float(
        (weights_v * transitions[labels[:-1, :], labels[1:, :]]).sum()
    )

    return energy


def graphcut_delta_energy(
    labels: np.ndarray,
    candidate: np.ndarray,
    unary: np.ndarray,
    transitions: np.ndarray,
    weights_h: np.ndarray,
    weights_v: np.ndarray,
) -> float:
    """
    Exact energy variation, evaluated on the modified pixels only.

    After the first cycle a move only touches a few thousand pixels : walking
    through the whole (H, W, nb_labels) array just to check it is wasteful.

    :returns: float energy difference
    """
    changed = candidate != labels
    if not changed.any():
        return 0.0

    rows, cols = np.nonzero(changed)
    delta = float(
        unary[rows, cols, candidate[rows, cols]].sum()
        - unary[rows, cols, labels[rows, cols]].sum()
    )

    for weights, slice_p, slice_q in (
        (
            weights_h,
            (slice(None), slice(0, -1)),
            (slice(None), slice(1, None)),
        ),
        (
            weights_v,
            (slice(0, -1), slice(None)),
            (slice(1, None), slice(None)),
        ),
    ):
        pairs = changed[slice_p] | changed[slice_q]
        if not pairs.any():
            continue

        rows, cols = np.nonzero(pairs)
        old_p, old_q = labels[slice_p], labels[slice_q]
        new_p, new_q = candidate[slice_p], candidate[slice_q]

        delta += float(
            (
                weights[rows, cols]
                * (
                    transitions[new_p[rows, cols], new_q[rows, cols]]
                    - transitions[old_p[rows, cols], old_q[rows, cols]]
                )
            ).sum()
        )

    return delta


def expansion_move(
    unary: np.ndarray,
    transitions: np.ndarray,
    weights_h: np.ndarray,
    weights_v: np.ndarray,
    labels: np.ndarray,
    alpha: int,
    active: np.ndarray = None,
    prepared_edges: tuple = None,
) -> np.ndarray:
    """
    Build and solve the binary problem of a single alpha move.

    :param np.ndarray active: boolean mask of the pixels allowed to change
    :param tuple prepared_edges: output of prepare_edges, rebuilt on the fly
        when not provided
    :returns: candidate labelling, or None if nothing can change
    """
    height, width = labels.shape

    if prepared_edges is None:
        prepared_edges = prepare_edges(weights_h, weights_v)

    is_node = labels != alpha
    if active is not None:
        is_node &= active

    nb_nodes = int(is_node.sum())
    if nb_nodes == 0:
        return None

    index = np.full((height, width), -1, np.int64)
    index[is_node] = np.arange(nb_nodes, dtype=np.int64)

    graph = maxflow.Graph[float]()
    node_ids = np.asarray(graph.add_nodes(nb_nodes), dtype=np.int64).ravel()

    cap_alpha = unary[is_node, alpha].astype(np.float64)
    cap_keep = unary[is_node, labels[is_node]].astype(np.float64)

    aux_ids, aux_caps = [], []

    for edge_w, slice_p, slice_q in prepared_edges:
        index_p, index_q = index[slice_p].ravel(), index[slice_q].ravel()
        label_p, label_q = labels[slice_p].ravel(), labels[slice_q].ravel()

        node_p, node_q = index_p >= 0, index_q >= 0

        # Boykov-Veksler-Zabih construction
        both = node_p & node_q
        if both.any():
            side_a, side_b = index_p[both], index_q[both]
            pair_p, pair_q = label_p[both], label_q[both]
            pair_w = edge_w[both]

            same = pair_p == pair_q
            if same.any():
                capacity = pair_w[same] * transitions[pair_p[same], alpha]
                graph.add_edges(side_a[same], side_b[same], capacity, capacity)

            different = ~same
            nb_aux = int(different.sum())
            if nb_aux:
                aux = np.asarray(
                    graph.add_nodes(nb_aux), dtype=np.int64
                ).ravel()
                cap_pa = (
                    pair_w[different] * transitions[pair_p[different], alpha]
                )
                cap_aq = (
                    pair_w[different] * transitions[alpha, pair_q[different]]
                )
                graph.add_edges(side_a[different], aux, cap_pa, cap_pa)
                graph.add_edges(aux, side_b[different], cap_aq, cap_aq)
                aux_ids.append(aux)
                aux_caps.append(
                    pair_w[different]
                    * transitions[pair_p[different], pair_q[different]]
                )

        # --- node / frozen pixel pairs : folded into the t-links of the node
        only_p = node_p & ~node_q
        if only_p.any():
            target, edge = index_p[only_p], edge_w[only_p]
            cap_keep += np.bincount(
                target,
                weights=edge * transitions[label_p[only_p], label_q[only_p]],
                minlength=nb_nodes,
            )
            cap_alpha += np.bincount(
                target,
                weights=edge * transitions[alpha, label_q[only_p]],
                minlength=nb_nodes,
            )

        only_q = node_q & ~node_p
        if only_q.any():
            target, edge = index_q[only_q], edge_w[only_q]
            cap_keep += np.bincount(
                target,
                weights=edge * transitions[label_p[only_q], label_q[only_q]],
                minlength=nb_nodes,
            )
            cap_alpha += np.bincount(
                target,
                weights=edge * transitions[label_p[only_q], alpha],
                minlength=nb_nodes,
            )

    graph.add_grid_tedges(node_ids, cap_alpha, cap_keep)

    if aux_ids:
        aux = np.concatenate(aux_ids)
        capacity = np.concatenate(aux_caps)
        graph.add_grid_tedges(aux, np.zeros_like(capacity), capacity)

    graph.maxflow()

    takes_alpha = graph.get_grid_segments(node_ids)
    if not takes_alpha.any():
        return None

    candidate = labels.copy()
    rows, cols = np.nonzero(is_node)
    candidate[rows[takes_alpha], cols[takes_alpha]] = alpha

    return candidate


def alpha_expansion(
    unary: np.ndarray,
    transitions: np.ndarray,
    weights_h: np.ndarray,
    weights_v: np.ndarray,
    initial_labels: np.ndarray = None,
    valid: np.ndarray = None,
    n_cycles: int = 6,
    alphas: list = None,
    band: int = None,
    initial_active: np.ndarray = None,
) -> tuple:
    """
    Minimise the energy by alpha-expansion.

    A move is only accepted if the energy actually decreases.

    :param np.ndarray initial_labels: starting labelling, argmin of the data
        term if None
    :param np.ndarray valid: boolean mask of the pixels allowed to change,
        invalid pixels keep their label for the whole optimisation
    :param list alphas: label indices allowed as alpha. By default only the
        ones present in the initialisation : a label absent from the mask costs
        one full max-flow per cycle for nothing.
    :param int band: if set, from the second cycle on only a band of that
        radius around the pixels modified during the previous cycle is
        reoptimised
    :param np.ndarray initial_active: restriction applied to the first cycle
        as well, defaults to valid
    :returns: (label indices, final energy)
    """
    height, width, _ = unary.shape
    unary = np.ascontiguousarray(unary, dtype=np.float32)

    if initial_labels is None:
        labels = unary.argmin(axis=2).astype(np.int32)
    else:
        labels = initial_labels.astype(np.int32).copy()

    if alphas is None:
        candidates = labels if valid is None else labels[valid]
        alphas = [int(a) for a in np.unique(candidates)]
    else:
        alphas = [int(a) for a in alphas]

    energy = graphcut_energy(labels, unary, transitions, weights_h, weights_v)

    remaining = set(alphas)
    active = valid if initial_active is None else initial_active
    structure = ndi.generate_binary_structure(2, 2)

    prepared_edges = prepare_edges(weights_h, weights_v)

    for _ in range(n_cycles):
        modified = np.zeros((height, width), bool)

        for alpha in alphas:
            if alpha not in remaining:
                continue

            candidate = expansion_move(
                unary,
                transitions,
                weights_h,
                weights_v,
                labels,
                alpha,
                active,
                prepared_edges,
            )
            if candidate is None:
                remaining.discard(alpha)
                continue

            delta = graphcut_delta_energy(
                labels, candidate, unary, transitions, weights_h, weights_v
            )
            if delta < -1e-6:
                modified |= candidate != labels
                labels = candidate
                energy += delta
                remaining = set(alphas) - {alpha}
            else:
                remaining.discard(alpha)

        if not remaining:
            break

        if band is not None:
            active = ndi.binary_dilation(modified, structure, iterations=band)
            if valid is not None:
                active &= valid

    return labels, energy


def downscale_problem(
    unary: np.ndarray,
    weights_h: np.ndarray,
    weights_v: np.ndarray,
    valid: np.ndarray,
    factor: int,
) -> tuple:
    """
    Aggregate the problem by a factor.

    :returns: (coarse unary, coarse horizontal weights, coarse vertical
        weights, coarse valid mask, coarse height, coarse width)
    """
    height, width, nb_labels = unary.shape
    coarse_h, coarse_w = height // factor, width // factor

    coarse_unary = (
        unary[: coarse_h * factor, : coarse_w * factor]
        .reshape(coarse_h, factor, coarse_w, factor, nb_labels)
        .sum((1, 3))
    )

    coarse_weights_h = (
        factor
        * weights_h[: coarse_h * factor, :]
        .reshape(coarse_h, factor, -1)
        .mean(1)[:, factor - 1 :: factor][:, : coarse_w - 1]
    )
    coarse_weights_v = (
        factor
        * weights_v[:, : coarse_w * factor]
        .reshape(-1, coarse_w, factor)
        .mean(2)[factor - 1 :: factor, :][: coarse_h - 1, :]
    )

    coarse_valid = None
    if valid is not None:
        coarse_valid = (
            valid[: coarse_h * factor, : coarse_w * factor]
            .reshape(coarse_h, factor, coarse_w, factor)
            .mean((1, 3))
            > 0.5
        )

    return (
        np.ascontiguousarray(coarse_unary, np.float32),
        np.ascontiguousarray(coarse_weights_h, np.float32),
        np.ascontiguousarray(coarse_weights_v, np.float32),
        coarse_valid,
        coarse_h,
        coarse_w,
    )


def alpha_expansion_multiscale(
    unary: np.ndarray,
    transitions: np.ndarray,
    weights_h: np.ndarray,
    weights_v: np.ndarray,
    initial_labels: np.ndarray,
    valid: np.ndarray = None,
    factor: int = 2,
    radius: int = None,
    n_cycles: int = 6,
    refine_band: int = 0,
    always_active: np.ndarray = None,
) -> tuple:
    """
    Coarse to fine alpha-expansion.

    :param int factor: downscaling factor, 0 or 1 disables the coarse stage
    :param int radius: radius of the band refined at full resolution
    :param int refine_band: if > 0, the first full resolution cycle is also
        restricted to a band of that radius around the boundaries of the
        upsampled coarse solution, plus always_active.
    :param np.ndarray always_active: pixels that must stay free to change even
        outside that band (shadows, undecided classes)
    :returns: (label indices, final energy)
    """
    if factor is None or factor <= 1:
        return alpha_expansion(
            unary,
            transitions,
            weights_h,
            weights_v,
            initial_labels=initial_labels,
            valid=valid,
            n_cycles=n_cycles,
        )

    height, width, nb_labels = unary.shape
    radius = factor + 2 if radius is None else radius

    if min(height, width) < 4 * factor:
        return alpha_expansion(
            unary,
            transitions,
            weights_h,
            weights_v,
            initial_labels=initial_labels,
            valid=valid,
            n_cycles=n_cycles,
        )

    (
        coarse_unary,
        coarse_weights_h,
        coarse_weights_v,
        coarse_valid,
        coarse_h,
        coarse_w,
    ) = downscale_problem(unary, weights_h, weights_v, valid, factor)

    # coarse initialisation = majority vote inside each block
    block = initial_labels[: coarse_h * factor, : coarse_w * factor].reshape(
        coarse_h, factor, coarse_w, factor
    )
    coarse_labels = (
        np.stack([(block == k).sum((1, 3)) for k in range(nb_labels)])
        .argmax(0)
        .astype(np.int32)
    )

    coarse_labels, _ = alpha_expansion(
        coarse_unary,
        transitions,
        coarse_weights_h,
        coarse_weights_v,
        initial_labels=coarse_labels,
        valid=coarse_valid,
        n_cycles=n_cycles,
    )

    labels = initial_labels.astype(np.int32).copy()
    upsampled = np.kron(coarse_labels, np.ones((factor, factor), np.int32))
    labels[: upsampled.shape[0], : upsampled.shape[1]] = upsampled

    if valid is not None:
        labels = np.where(valid, labels, initial_labels)

    active = valid
    if refine_band:
        active = boundary_band(labels, int(refine_band))
        if always_active is not None:
            active |= always_active
        if valid is not None:
            active &= valid

    return alpha_expansion(
        unary,
        transitions,
        weights_h,
        weights_v,
        initial_labels=labels,
        valid=valid,
        n_cycles=n_cycles,
        band=radius,
        initial_active=active,
    )


# --- Hierarchical hole filling ----------------------------------------------


def fill_holes(
    im_classif: np.ndarray,
    target: list,
    absorbable: list,
    max_area: int,
    max_area_shadow: int = None,
    shadow: np.ndarray = None,
    closing_radius: int = 0,
    vote_size: int = 15,
) -> np.ndarray:
    """
    Fill the holes of one class group.

    A hole is filled only if it is entirely enclosed in the target class, if
    its area is below the threshold and if its current class is allowed to be
    absorbed. The area threshold is raised for holes that are mostly in shadow.
    The hole is filled with the target sub-class dominating its neighbourhood.

    :param np.ndarray im_classif: mask holding nomenclature codes
    :param list target: classes forming the region to complete
    :param list absorbable: classes allowed to be replaced
    :param int max_area: maximum hole area, in pixels
    :param int max_area_shadow: maximum hole area for shadowed holes
    :param np.ndarray shadow: boolean shadow mask
    :param int closing_radius: optional morphological closing before the hole
        detection
    :param int vote_size: window size of the local vote
    :returns: updated mask
    """
    max_area_shadow = max_area if max_area_shadow is None else max_area_shadow

    region = np.isin(im_classif, target)

    if closing_radius:
        grid_y, grid_x = np.mgrid[
            -closing_radius : closing_radius + 1,
            -closing_radius : closing_radius + 1,
        ]
        element = (grid_y**2 + grid_x**2) <= closing_radius**2
        region = ndi.binary_closing(region, element)

    holes = ndi.binary_fill_holes(region) & ~region
    if not holes.any():
        return im_classif

    labelled, nb_holes = ndi.label(
        holes, structure=ndi.generate_binary_structure(2, 2)
    )
    areas = np.bincount(labelled.ravel(), minlength=nb_holes + 1).astype(float)

    if shadow is not None:
        shadowed = np.bincount(
            labelled.ravel(),
            weights=shadow.ravel().astype(float),
            minlength=nb_holes + 1,
        )
        shadow_fraction = shadowed / np.maximum(areas, 1)
    else:
        shadow_fraction = np.zeros(nb_holes + 1)

    keep = (areas <= max_area) | (
        (shadow_fraction > 0.5) & (areas <= max_area_shadow)
    )
    keep[0] = False

    to_fill = keep[labelled] & np.isin(im_classif, absorbable)
    if not to_fill.any():
        return im_classif

    # Local vote to pick the filling sub-class
    rows, cols = np.nonzero(to_fill)
    row_min = max(int(rows.min()) - vote_size, 0)
    row_max = min(int(rows.max()) + vote_size + 1, im_classif.shape[0])
    col_min = max(int(cols.min()) - vote_size, 0)
    col_max = min(int(cols.max()) + vote_size + 1, im_classif.shape[1])

    window = (slice(row_min, row_max), slice(col_min, col_max))
    sub_classif = im_classif[window]

    scores = [
        ndi.uniform_filter((sub_classif == code).astype(np.float32), vote_size)
        for code in target
    ]
    choice = np.array(target)[np.argmax(np.stack(scores), axis=0)]

    filled = im_classif.copy()
    sub_fill = to_fill[window]
    filled[window][sub_fill] = choice[sub_fill]

    return filled


def fill_holes_hierarchical(
    im_classif: np.ndarray,
    shadow: np.ndarray = None,
    max_area: int = 200,
    max_area_shadow: int = 2000,
    closing_radius: int = 0,
) -> np.ndarray:
    """
    Fill holes class by class, by decreasing priority.

    High vegetation first, then low vegetation, then the smooth undefined
    class. This complements the graph cut : the regularisation closes the small
    gaps, this closes the large ones it cannot reach.

    :param np.ndarray im_classif: mask holding nomenclature codes
    :param np.ndarray shadow: boolean shadow mask
    :param int max_area: maximum hole area, in pixels
    :param int max_area_shadow: maximum hole area for shadowed holes
    :param int closing_radius: optional morphological closing
    :returns: updated mask
    """
    steps = [
        (
            [
                VEG_CODE + MIDDLE_TEXTURE_CODE,
                VEG_CODE + HIGH_TEXTURE_CODE,
            ],
            [
                NO_VEG_CODE,
                UNDEFINED_VEG + LOW_TEXTURE_CODE,
                UNDEFINED_VEG + MIDDLE_TEXTURE_CODE,
                UNDEFINED_VEG + HIGH_TEXTURE_CODE,
                VEG_CODE + LOW_TEXTURE_CODE,
            ],
        ),
        (
            [VEG_CODE + LOW_TEXTURE_CODE],
            [
                NO_VEG_CODE,
                UNDEFINED_VEG + LOW_TEXTURE_CODE,
                UNDEFINED_VEG + MIDDLE_TEXTURE_CODE,
                UNDEFINED_VEG + HIGH_TEXTURE_CODE,
            ],
        ),
        (
            [UNDEFINED_VEG + LOW_TEXTURE_CODE],
            [
                NO_VEG_CODE,
                UNDEFINED_VEG + MIDDLE_TEXTURE_CODE,
                UNDEFINED_VEG + HIGH_TEXTURE_CODE,
            ],
        ),
    ]

    for target, absorbable in steps:
        im_classif = fill_holes(
            im_classif,
            target,
            absorbable,
            max_area,
            max_area_shadow,
            shadow,
            closing_radius,
        )

    return im_classif


# --- SLURP tasks ------------------------------------------------------------


def graphcut_stats_task(
    im_vhr: np.ndarray,
    im_ndvi: np.ndarray,
    im_texture: np.ndarray,
    valid_stack: np.ndarray,
    visible_bands: tuple,
    ndvi_scale: float,
) -> list:
    """
    Accumulate the global statistics of the graph cut attributes.

    The contrast term normalizes luminance, NDVI and texture, and the shadow
    mask thresholds the luminance. Computing those statistics per tile would
    make the regularisation tile dependent and leave visible seams, so they are
    reduced over the whole image first.

    :param np.ndarray im_vhr: VHR image tile (bands, H, W)
    :param np.ndarray im_ndvi: NDVI tile
    :param np.ndarray im_texture: texture tile
    :param np.ndarray valid_stack: validity mask (0 = valid pixel)
    :param tuple visible_bands: indices of the visible bands in the VHR image
    :param float ndvi_scale: factor converting the stored NDVI to [-1, 1]
    :returns: [accumulator array]
    """
    valid = np.squeeze(valid_stack) == 0
    if not valid.any():
        return [np.zeros(7, dtype=np.float64)]

    if im_vhr.ndim == 2:
        luminance = im_vhr.astype(np.float32)
    else:
        luminance = im_vhr[list(visible_bands)].astype(np.float32).mean(axis=0)

    ndvi = np.squeeze(im_ndvi).astype(np.float32) / ndvi_scale
    texture = np.squeeze(im_texture).astype(np.float32)

    accumulator = np.zeros(7, dtype=np.float64)
    accumulator[0] = float(np.count_nonzero(valid))

    for offset, layer in enumerate((luminance, ndvi, texture)):
        accumulator[1 + 2 * offset] = float(
            np.sum(layer, dtype=np.float64, where=valid)
        )
        accumulator[2 + 2 * offset] = float(
            np.sum(np.multiply(layer, layer), dtype=np.float64, where=valid)
        )

    return [accumulator]


def graphcut_stats_concatenate(chunks_output_scalars: list) -> list:
    """
    Sum the accumulators coming from every sub-tile.

    :param list chunks_output_scalars: list of per-chunk accumulators
    :returns: [global accumulator]
    """
    total = np.array(chunks_output_scalars[0][0], copy=True)

    for chunk in chunks_output_scalars[1:]:
        total += chunk[0]

    return [total]


def graphcut_moments(accumulator: np.ndarray) -> dict:
    """
    Turn the reduced accumulator into means and standard deviations.

    :param np.ndarray accumulator: [count, sum, sumsq] x 3 layers
    :returns: dict of global moments
    """
    count = max(float(accumulator[0]), 1.0)
    moments = {}

    for offset, name in enumerate(("luminance", "ndvi", "texture")):
        mean = accumulator[1 + 2 * offset] / count
        variance = accumulator[2 + 2 * offset] / count - mean**2
        moments[f"{name}_mean"] = float(mean)
        moments[f"{name}_std"] = float(np.sqrt(max(variance, 0.0))) + EPS

    return moments


def graphcut_task(
    im_classif: np.ndarray,
    im_ndvi: np.ndarray,
    im_texture: np.ndarray,
    im_vhr: np.ndarray,
    valid_stack: np.ndarray,
    im_shadow: np.ndarray,
    moments: dict,
    positions: np.ndarray,
    transitions: np.ndarray,
    confusion: np.ndarray,
    visible_bands: tuple,
    ndvi_scale: float,
    ndvi_threshold: float,
    ndvi_slope: float,
    ndvi_weight: float,
    shadow_factor: float,
    attribute_weights: tuple,
    contrast_sigma: float,
    contrast_kernel: str,
    diffusion_iterations: int,
    regularisation: float,
    lambda_absolute: float,
    scale_factor: int,
    n_cycles: int,
    fill_holes_enabled: bool,
    hole_max_area: int,
    hole_max_area_shadow: int,
    refine_band: int = 0,
) -> np.ndarray:
    """
    Refine one tile of the vegetation mask by alpha-expansion.

    :param np.ndarray im_classif: input mask tile, nomenclature codes
    :param np.ndarray im_ndvi: NDVI tile
    :param np.ndarray im_texture: texture tile
    :param np.ndarray im_vhr: VHR image tile (bands, H, W)
    :param np.ndarray valid_stack: validity mask (0 = valid pixel)
    :param np.ndarray im_shadow: shadow mask tile (2 = confirmed shadow)
    :param dict moments: global moments of the attributes
    :param np.ndarray positions: label embedding
    :param np.ndarray transitions: transition matrix V
    :param np.ndarray confusion: prior confusion matrix M
    :param float regularisation: relative weight of the regularisation,
        calibrated on the dynamic of the tile
    :param float lambda_absolute: if set, used as the absolute regularisation
        weight and the calibration is skipped
    :param int scale_factor: coarse to fine downscaling factor
    :returns: refined mask tile
    """
    im_classif = np.squeeze(im_classif)
    valid = np.squeeze(valid_stack) == 0

    if not valid.any():
        return im_classif.astype(np.uint8)

    ndvi = np.squeeze(im_ndvi).astype(np.float32) / ndvi_scale
    texture = np.squeeze(im_texture).astype(np.float32)

    if im_vhr.ndim == 2:
        luminance = im_vhr.astype(np.float32)
    else:
        luminance = im_vhr[list(visible_bands)].astype(np.float32).mean(axis=0)
    luminance = ndi.gaussian_filter(luminance, 1.0)

    # ------------------------------------------------------------------
    # DATA TERM
    # ------------------------------------------------------------------

    observed, known = observed_labels(im_classif)
    unary = unary_cost(observed, confusion)
    unary = add_ndvi_evidence(
        unary, ndvi, positions, ndvi_threshold, ndvi_slope, ndvi_weight
    )

    shadow = np.squeeze(np.asarray(im_shadow)) == 2
    unary = weaken_unaries(unary, shadow & valid, shadow_factor)

    uncertain = (
        observed >= GRAPHCUT_INDEX[UNDEFINED_VEG + LOW_TEXTURE_CODE]
    ) & (observed <= GRAPHCUT_INDEX[UNDEFINED_VEG + HIGH_TEXTURE_CODE])
    uncertain |= shadow

    # invalid pixels contribute nothing and never change
    unary[~valid] = 0.0

    # ------------------------------------------------------------------
    # REGULARISATION TERM
    # ------------------------------------------------------------------

    if diffusion_iterations > 0:
        smoothed_ndvi = anisotropic_diffusion(
            ndvi, gamma=0.1, max_iter=diffusion_iterations, sigma=0.3
        )
    else:
        smoothed_ndvi = ndi.gaussian_filter(ndvi, 1.0)

    attributes = [
        normalize_with(
            smoothed_ndvi, moments["ndvi_mean"], moments["ndvi_std"]
        ),
        normalize_with(
            luminance, moments["luminance_mean"], moments["luminance_std"]
        ),
        normalize_with(
            texture, moments["texture_mean"], moments["texture_std"]
        ),
    ]

    weights_h = contrast_weights(
        attributes,
        attribute_weights,
        axis=1,
        sigma=contrast_sigma,
        kernel=contrast_kernel,
    )
    weights_v = contrast_weights(
        attributes,
        attribute_weights,
        axis=0,
        sigma=contrast_sigma,
        kernel=contrast_kernel,
    )

    # no regularisation across an invalid pixel
    weights_h *= (valid[:, :-1] & valid[:, 1:]).astype(np.float32)
    weights_v *= (valid[:-1, :] & valid[1:, :]).astype(np.float32)


    if lambda_absolute is not None:
        lambda_reg = float(lambda_absolute)
    else:
        amplitude = float(np.mean(unary.max(axis=2) - unary.min(axis=2)))
        mean_weight = float(
            np.concatenate([weights_h.ravel(), weights_v.ravel()]).mean()
        )
        mean_transition = float(transitions[transitions > 0].mean())
        lambda_reg = (
            regularisation * amplitude / (mean_weight * mean_transition + EPS)
        )

    weights_h = weights_h * lambda_reg
    weights_v = weights_v * lambda_reg

    # ------------------------------------------------------------------
    # ALPHA-EXPANSION
    # ------------------------------------------------------------------

    labels, _ = alpha_expansion_multiscale(
        unary,
        transitions,
        weights_h,
        weights_v,
        initial_labels=observed,
        valid=valid,
        factor=scale_factor,
        n_cycles=n_cycles,
        refine_band=refine_band,
        always_active=uncertain,
    )

    refined = GRAPHCUT_LABELS[labels].astype(np.uint8)

    # ------------------------------------------------------------------
    # HIERARCHICAL HOLE FILLING
    # ------------------------------------------------------------------

    if fill_holes_enabled:
        refined = fill_holes_hierarchical(
            refined,
            shadow=shadow & valid,
            max_area=hole_max_area,
            max_area_shadow=hole_max_area_shadow,
        )

    # pixels outside the nomenclature and invalid pixels are left untouched
    refined = np.where(known & valid, refined, im_classif)
    refined = np.where(valid, refined, NODATA_INT8)

    return refined.astype(np.uint8)


def graphcut_refinement(
    args: argparse.Namespace,
    slurp_manager: slurpContextManager,
    final_mask: list,
    key_ndvi: list,
    key_texture: list,
    key_vhr: list,
    key_valid_stack: list,
    key_shadowmask: list,
    output_profile: dict,
):
    """
    Refine the vegetation mask by alpha-expansion, using the SLURP framework.

    Two passes : the first one reduces the global statistics of the attributes
    over the whole image, the second one runs the graph cut tile by tile with a
    stable margin. Tiles are independent once the margin is fixed, so the whole
    refinement is parallelised by the SLURP executor.

    :param argparse.Namespace args: runtime configuration
    :param slurpContextManager slurp_manager: SLURP execution context
    :param list final_mask: keys of the mask to refine
    :param list key_ndvi: NDVI keys
    :param list key_texture: texture keys
    :param list key_vhr: VHR image keys
    :param list key_valid_stack: valid stack keys
    :param list key_shadowmask: shadow mask keys (2 = confirmed shadow)
    :param dict output_profile: output raster profile
    :returns: keys of the refined mask
    """
    if not args.graphcut:
        return final_mask

    if maxflow is None:
        logger.warning(
            "PyMaxflow is not available, skipping the graph cut refinement "
            "(pip install PyMaxflow)"
        )
        return final_mask

    logger.info("Graph cut refinement of the vegetation mask...")

    # ======================================================
    # LABEL MODEL
    # ======================================================

    positions = build_label_positions(args.graphcut_texture_spread)
    transitions = transition_matrix(positions, args.graphcut_truncation)
    confusion = prior_confusion_matrix(
        positions, GRAPHCUT_CONFIDENCE, args.graphcut_tau
    )

    checks = check_metric(transitions)
    if not all(checks.values()):
        logger.warning(
            f"Transition matrix is not a metric ({checks}), "
            "alpha-expansion would be invalid, skipping the refinement"
        )
        return final_mask

    logger.debug(f"Transition matrix:\n{np.round(transitions, 2)}")
    logger.debug(f"Prior confusion matrix:\n{np.round(confusion, 3)}")

    # ======================================================
    # GLOBAL STATISTICS
    # ======================================================

    accumulator = mp_n_to_m_scalars(
        inputs=[
            key_vhr[0][0],
            key_ndvi[0][0],
            key_texture[0][0],
            key_valid_stack[0][0],
        ],
        image_height=output_profile["height"],
        image_width=output_profile["width"],
        func=graphcut_stats_task,
        func_parameters={
            "visible_bands": tuple(args.graphcut_visible_bands),
            "ndvi_scale": float(args.graphcut_ndvi_scale),
        },
        context_manager=slurp_manager,
        reducer=graphcut_stats_concatenate,
    )

    moments = graphcut_moments(accumulator[0])
    logger.debug(f"Global attribute moments: {moments}")

    if moments["luminance_std"] <= EPS:
        logger.warning(
            "Degenerate luminance statistics, skipping the refinement"
        )
        return final_mask

    # ======================================================
    # NDVI SIGMOID THRESHOLD
    # ======================================================

    ndvi_threshold = args.graphcut_ndvi_threshold
    if ndvi_threshold is None:
        min_ndvi_veg = getattr(args, "min_ndvi_veg", None)
        if min_ndvi_veg is None:
            ndvi_threshold = 0.25
        else:
            ndvi_threshold = float(min_ndvi_veg) / float(
                args.graphcut_ndvi_scale
            )
    logger.debug(f"NDVI sigmoid centred on {ndvi_threshold=}")

    # ======================================================
    # STABLE MARGIN
    # ======================================================

    margin = max(
        int(args.graphcut_margin),
        4 * int(args.graphcut_scale_factor),
    )
    if args.graphcut_fill_holes:
        # only the hole filling needs a margin proportional to the hole size
        margin = max(
            margin, ceil(sqrt(max(args.graphcut_hole_max_area_shadow, 1)))
        )

    # ======================================================
    # SLURP EXECUTION
    # ======================================================

    refined_mask = mp_n_to_m_images(
        inputs=[
            final_mask[0],
            key_ndvi[0][0],
            key_texture[0][0],
            key_vhr[0][0],
            key_valid_stack[0][0],
            key_shadowmask[0][0],
        ],
        image_height=output_profile["height"],
        image_width=output_profile["width"],
        output_profiles=[output_profile],
        output_keys=["graphcut"],
        func=graphcut_task,
        func_parameters={
            "moments": moments,
            "positions": positions,
            "transitions": transitions,
            "confusion": confusion,
            "visible_bands": tuple(args.graphcut_visible_bands),
            "ndvi_scale": float(args.graphcut_ndvi_scale),
            "ndvi_threshold": float(ndvi_threshold),
            "ndvi_slope": float(args.graphcut_ndvi_slope),
            "ndvi_weight": float(args.graphcut_ndvi_weight),
            "shadow_factor": float(args.graphcut_shadow_factor),
            "attribute_weights": tuple(args.graphcut_attribute_weights),
            "contrast_sigma": float(args.graphcut_contrast_sigma),
            "contrast_kernel": str(args.graphcut_contrast_kernel),
            "diffusion_iterations": int(args.graphcut_diffusion_iterations),
            "regularisation": float(args.graphcut_lambda),
            "lambda_absolute": (
                None
                if args.graphcut_lambda_absolute is None
                else float(args.graphcut_lambda_absolute)
            ),
            "scale_factor": int(args.graphcut_scale_factor),
            "refine_band": int(args.graphcut_refine_band),
            "n_cycles": int(args.graphcut_cycles),
            "fill_holes_enabled": bool(args.graphcut_fill_holes),
            "hole_max_area": int(args.graphcut_hole_max_area),
            "hole_max_area_shadow": int(args.graphcut_hole_max_area_shadow),
        },
        context_manager=slurp_manager,
        stable_margin=margin,
        binary=True,
    )

    if args.save_mode in ["all", "debug"]:
        output_path = args.vegetationmask.replace(".tif", "_graphcut.tif")
        slurp_manager.write_tif(
            data=refined_mask[0],
            path=output_path,
            target_profile=output_profile,
        )

    return refined_mask


def display_infos(
    args,
    end_time,
    t0,
    time_closing,
    time_cluster,
    time_final,
    time_graphcut,
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
        "- Graph cut refinement  :\t"
        + utils.convert_time(time_graphcut - time_final)
    )
    logger.info(
        "- Post-processing       :\t"
        + utils.convert_time(time_closing - time_graphcut)
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
        "--version",
        default=None,
        action="version",
        version=f"SLURP {__version__}",
    )

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

    group_gc = parser.add_argument_group(description="*** GRAPH CUT ***")
    group_gc.add_argument(
        "-graphcut",
        default=None,
        action="store_true",
        help="Refine the vegetation mask with a multi-label graph cut "
        "(alpha-expansion) before the morphological post-processing",
    )
    group_gc.add_argument(
        "-graphcut_lambda",
        type=float,
        help="Relative weight of the regularisation against the data term",
    )
    group_gc.add_argument(
        "-graphcut_truncation",
        type=float,
        help="Upper bound of the semantic transition cost (0-1)",
    )
    group_gc.add_argument(
        "-graphcut_tau",
        type=float,
        help="Spread of the prior confusion over the neighbouring classes "
        "(small value = confusions between close classes only)",
    )
    group_gc.add_argument(
        "-graphcut_texture_spread",
        type=float,
        help="Extent of the texture axis relative to the vegetation axis "
        "in the label embedding",
    )
    group_gc.add_argument(
        "-graphcut_ndvi_scale",
        type=float,
        help="Factor converting the stored NDVI to physical values in [-1, 1]",
    )
    group_gc.add_argument(
        "-graphcut_ndvi_threshold",
        type=float,
        help="NDVI value at the centre of the vegetation sigmoid "
        "(defaults to min_ndvi_veg)",
    )
    group_gc.add_argument(
        "-graphcut_ndvi_slope",
        type=float,
        help="Width of the vegetation sigmoid, in physical NDVI units",
    )
    group_gc.add_argument(
        "-graphcut_ndvi_weight",
        type=float,
        help="Weight of the NDVI evidence in the data term (0 disables it)",
    )
    group_gc.add_argument(
        "-graphcut_shadow_factor",
        type=float,
        help="Multiplicative factor applied to the data term in shadows "
        "(small value = the label is decided by the neighbourhood)",
    )
    group_gc.add_argument(
        "-graphcut_lambda_absolute",
        type=float,
        help="Absolute regularisation weight, bypassing the per-tile "
        "calibration of graphcut_lambda",
    )
    group_gc.add_argument(
        "-graphcut_contrast_sigma",
        type=float,
        help="Contrast scale of the regularisation kernel, in normalized "
        "attribute units",
    )
    group_gc.add_argument(
        "-graphcut_contrast_kernel",
        type=str,
        choices=["gauss", "cauchy"],
        help="Regularisation kernel : gauss decreases fast (all or nothing), "
        "cauchy keeps some regularisation across medium contrasts",
    )
    group_gc.add_argument(
        "-graphcut_diffusion_iterations",
        type=int,
        help="Number of anisotropic diffusion iterations applied to the NDVI "
        "before computing the contrast (0 disables the diffusion)",
    )
    group_gc.add_argument(
        "-graphcut_scale_factor",
        type=int,
        help="Coarse to fine downscaling factor (1 disables the coarse stage)",
    )
    group_gc.add_argument(
        "-graphcut_refine_band",
        type=int,
        help="Radius, in pixels, of the band reoptimised at full resolution "
        "around the boundaries of the coarse solution (0 : optimise every "
        "pixel, slower and usually identical)",
    )
    group_gc.add_argument(
        "-graphcut_cycles",
        type=int,
        help="Maximum number of alpha-expansion cycles",
    )
    group_gc.add_argument(
        "-graphcut_margin",
        type=int,
        help="Stable margin of the graph cut tiles, in pixels",
    )
    group_gc.add_argument(
        "-graphcut_fill_holes",
        default=None,
        action="store_true",
        help="Apply the hierarchical hole filling after the graph cut",
    )
    group_gc.add_argument(
        "-graphcut_hole_max_area",
        type=int,
        help="Maximum area, in pixels, of a hole filled after the graph cut",
    )
    group_gc.add_argument(
        "-graphcut_hole_max_area_shadow",
        type=int,
        help="Maximum area, in pixels, of a shadowed hole filled after the "
        "graph cut",
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


def slurp_vegetationmask(
    main_config: str,
    debug: bool,
    version: bool,
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
    graphcut: bool = None,
    graphcut_lambda: float = None,
    graphcut_lambda_absolute: float = None,
    graphcut_truncation: float = None,
    graphcut_tau: float = None,
    graphcut_texture_spread: float = None,
    graphcut_ndvi_scale: float = None,
    graphcut_ndvi_threshold: float = None,
    graphcut_ndvi_slope: float = None,
    graphcut_ndvi_weight: float = None,
    graphcut_shadow_factor: float = None,
    graphcut_contrast_sigma: float = None,
    graphcut_contrast_kernel: str = None,
    graphcut_diffusion_iterations: int = None,
    graphcut_scale_factor: int = None,
    graphcut_refine_band: int = None,
    graphcut_cycles: int = None,
    graphcut_margin: int = None,
    graphcut_fill_holes: bool = None,
    graphcut_hole_max_area: int = None,
    graphcut_hole_max_area_shadow: int = None,
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

    # Graph cut parameters are optional : fill in the ones the configuration
    # does not define, so that existing JSON files keep working unchanged
    set_graphcut_defaults(argsdict)

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

    with slurpContextManager(
        params, tile_mode=True, tile_max_size=args.tile_max_size
    ) as slurp_manager:

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
                shadowmask,
                vhr_profile,
                valid_stack_profile,
                ndvi_profile,
            ) = build_stack_vegetation(args, slurp_manager)

            time_stack = time.time()

            # =====================================================
            # SEGMENTATION
            # =====================================================

            logger.info("[1] Step: Segmentation")

            segments = segmentation(
                args, slurp_manager, ndvi, valid_stack, ndvi_profile
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
                ndvi_profile,
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

            # index of the first cluster labelled as vegetation, clipped so
            # that an inconsistent nb_clusters_veg cannot raise an IndexError
            index_first_veg = -min(
                max(int(args.nb_clusters_veg), 1), len(sorted_ndvi_centroids)
            )
            logger.debug(
                f"NDVI of 1st vegetation cluster {sorted_ndvi_centroids[index_first_veg]=}"
            )

            if args.autolabel:
                clusters_veg = vegetation_labeling_with_LCM(
                    vars(args), pred_veg
                )
            else:
                clusters_veg = vegetation_labeling_with_rule_of_third(
                    vars(args), pred_veg
                )

            pred_texture, sorted_texture_centroids = clustering_texture(
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
            # uint8 : the nomenclature never exceeds 23 and this table is
            # shipped to every worker for every tile
            final_clusters = np.zeros(size_result, dtype=np.uint8)
            final_clusters[as_bool_mask(mask_valid_indices)] = clusters

            final_mask = mp_n_to_m_images(
                inputs=[segments[0], valid_stack[0][0]],
                image_height=vhr_profile["height"],
                image_width=vhr_profile["width"],
                output_profiles=[eo_utils.single_uint8_profile([vhr_profile])],
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
                    target_profile=eo_utils.single_uint8_profile([vhr_profile]),
                )

            time_final = time.time()

            # An explicit min_ndvi_veg coming from the configuration is no
            # longer silently overwritten by the centroid of the clustering.
            if getattr(args, "min_ndvi_veg", None) is None:
                args.min_ndvi_veg = sorted_ndvi_centroids[index_first_veg]
            logger.debug(f"NDVI filter threshold: {args.min_ndvi_veg}")

            # =====================================================
            # GRAPH CUT REFINEMENT
            # =====================================================

            logger.info("[6] Step: Graph cut refinement")

            final_mask = graphcut_refinement(
                args,
                slurp_manager,
                final_mask,
                ndvi,
                texture,
                vhr,
                valid_stack,
                shadowmask,
                eo_utils.single_uint8_profile([vhr_profile]),
            )

            time_graphcut = time.time()

            # =====================================================
            # POSTPROCESS
            # =====================================================

            logger.info("[7] Step: Postprocess")

            final_mask = postprocess(
                args,
                slurp_manager,
                final_mask,
                valid_stack,
                ndvi,
                eo_utils.single_uint8_profile([vhr_profile]),
            )

            time_closing = time.time()
            # =====================================================
            # WRITE OUTPUT
            # =====================================================

            slurp_manager.write_tif(
                data=final_mask[0],
                path=args.vegetationmask,
                target_profile=eo_utils.single_uint8_profile([vhr_profile]),
            )

            t1 = time.time()

            display_infos(
                args,
                t1,
                t0,
                time_closing,
                time_cluster,
                time_final,
                time_graphcut,
                time_seg,
                time_stack,
                time_stats,
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
