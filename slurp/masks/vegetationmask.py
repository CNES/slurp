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
from math import ceil, sqrt

import eoscale.eo_executors as eoexe
import eoscale.manager as eom
import numpy as np
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
    return np.array([map_centroids[n] for n in pred])


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
    input_buffers: list, input_profiles: list, params: dict
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


def concat_seg(global_seg, tile_seg, tile, current_offset):
    """
    Merge one tile segmentation into global segmentation
    with globally unique labels.

    Parameters
    ----------
    global_seg : np.ndarray
        Final segmentation image (modified in-place)
    tile_seg : np.ndarray
        Segmentation result of the tile
    tile : Tile
        Tile spatial definition
    current_offset : int
        Current global label offset

    Returns
    -------
    int
        Updated offset
    """

    ys = slice(tile.start_y, tile.end_y + 1)
    xs = slice(tile.start_x, tile.end_x + 1)

    seg = tile_seg.copy()

    mask = seg > 0
    seg[mask] += current_offset

    global_seg[:, ys, xs] = np.where(mask, seg, 0)

    # compute next offset ONLY from tile
    next_offset = current_offset + seg.max()

    return next_offset


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
    nb_primitives = 3  # NDVI, NDWI, Texture

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
    Reduce partial statistics from tiles.
    """

    sums = np.stack([c[0] for c in chunks_output_scalars], axis=0)
    counts = np.stack([c[1] for c in chunks_output_scalars], axis=0)

    global_sum = sums.sum(axis=0)
    global_count = counts.sum(axis=0)

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


def finalize_task(input_buffers: list, input_profiles: list, params: dict):
    """
    Finalize mask : for each pixel in input segmentation,
    return class (low / high vegetation, etc.)

    :param list input_buffers: [segments, valid_stack]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: {"data": clusters} with clusters an array
    :returns: final mask
    """
    clustering = params["data"]
    # Load Cython module and launch C++ function
    ts_stats = ts.PyStats()

    final_mask = ts_stats.finalize(input_buffers[0], clustering)

    # Add nodata in final_mask (input_buffers[1] : valid mask)
    final_mask = np.where(input_buffers[1][0] == 0, final_mask, NODATA_INT8)

    return final_mask


def clean_task(
    input_buffers: list, input_profiles: list, params: dict
) -> np.ndarray:
    """
    Post-processing : remove small holes/objects, apply binary dilation on low veg
    and filter with the NDVI of the fist vegetation cluster

    :param list input_buffers: [final_seg, valid_stack, ndvi]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: final mask
    """
    im_classif = input_buffers[0][0]
    valid_stack = input_buffers[1][0]
    im_ndvi = input_buffers[2][0]

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

    # Filter final mask with a NDVI threshold (1st cluster of vegetation)
    # TODO : replace 0 by UNDEFINED_VEG ?
    im_classif = np.where(
        im_classif == LOW_VEG_CLASS,
        np.where(im_ndvi > params["min_ndvi_veg"], LOW_VEG_CLASS, 0),
        im_classif,
    )
    # TODO : replace 0 by UNDEFINED_VEG + MIDDLE_TEXTURE_CODE
    im_classif = np.where(
        im_classif > LOW_VEG_CLASS,
        np.where(
            im_ndvi > params["min_ndvi_veg"], VEG_CODE + MIDDLE_TEXTURE_CODE, 0
        ),
        im_classif,
    )

    im_classif = np.where(valid_stack == 0, im_classif, NODATA_INT8)

    return im_classif


def segmentation(
    args,
    slurp_manager,
    key_ndvi,
    key_valid_stack,
    ndvi_profile,
):

    logger.info("Segmentation processing...")

    input_keys = [
        key_ndvi[0][0],
        key_valid_stack[0][0],
    ]

    input_profile = deepcopy(ndvi_profile)

    output_profile = eo_utils.single_int32_profile(
        [deepcopy(ndvi_profile)]
    )

    # -------------------------------
    # RUN PARALLEL SEGMENTATION
    # -------------------------------
    future_seg = mp_n_to_m_images(
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
        stable_margin=0,
        binary=False,
    )

    # ==========================================================
    # POST PROCESS ? GLOBAL RELABELING
    # ==========================================================

    logger.info("Re-label segmentation globally...")

    seg = future_seg[0]

    unique_vals = np.unique(seg)
    unique_vals = unique_vals[unique_vals > 0]

    # Fast relabeling
    new_labels = np.arange(1, len(unique_vals) + 1)

    lut = np.zeros(unique_vals.max() + 1, dtype=np.int32)
    lut[unique_vals] = new_labels

    seg[:] = lut[seg]

    # ==========================================================
    # SAVE
    # ==========================================================

    output_path = args.vegetationmask.replace(".tif", "_slic.tif")

    slurp_manager.write_tif(
        data=seg,
        path=output_path,
        target_profile=output_profile,
    )

    return [seg]


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


def postprocess(args, eoscale_manager, final_seg, key_valid_stack, key_ndvi):
    """
    Performs morphological closing and other post-processing operations
    (binary dilation, removal of small objects, and holes,...)
    in the segmented image if the texture mode is enabled.

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
            inputs=[final_seg[0], key_valid_stack, key_ndvi],
            image_filter=clean_task,
            filter_parameters=vars(args),
            generate_output_profiles=eo_utils.single_uint8_profile,
            stable_margin=margin,
            context_manager=eoscale_manager,
            multiproc_context=args.multiproc_context,
            filter_desc="Post-processing...",
        )
    return final_seg


def process_stats(
    args,
    eoscale_manager,
    future_seg,
    key_ndvi,
    key_ndwi,
    key_texture,
    size_result,
    mask_valid_indices,
):
    """
    Computes statistics (mean NDVI, NDWI, and texture) for each segmented region.
    Then, the statistics are processed to generate data for clustering or classification.
    """
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

    mean_ndvi = stats[0][:size_result]
    mean_ndvi[np.where(mask_valid_indices == 0)] = NODATA_INT16
    mean_ndvi[np.where(mask_valid_indices)] = (
        mean_ndvi[np.where(mask_valid_indices)]
        / stats[1][np.where(mask_valid_indices)]
    )

    mean_ndwi = stats[0][size_result : 2 * size_result]
    mean_ndwi[np.where(mask_valid_indices == 0)] = NODATA_INT16
    mean_ndwi[np.where(mask_valid_indices)] = (
        mean_ndwi[np.where(mask_valid_indices)]
        / stats[1][np.where(mask_valid_indices)]
    )

    mean_texture = stats[0][2 * size_result : 3 * size_result]
    mean_texture[np.where(mask_valid_indices == 0)] = NODATA_INT16
    mean_texture[np.where(mask_valid_indices)] = (
        mean_texture[np.where(mask_valid_indices)]
        / stats[1][np.where(mask_valid_indices)]
    )

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

            key_ndvi, key_ndwi, key_vhr, key_texture, key_valid_stack = (
                build_stack(args, eoscale_manager)
            )

            time_stack = time.time()

            # Segmentation #

            future_seg = segmentation(
                args, eoscale_manager, key_ndvi, key_valid_stack
            )

            time_seg = time.time()

            # Stats #
            """
            *** Recover number total of segments and check valid segments ***
            res_seg contains segments from 1 to n
            - 0 stands for NO_DATA
            - 1 to n are different segments detected by SLIC
            - but some segments 'i' disappear from res_seg because they have been invalidated

            => we need to produce a mask of valid indices : 
            1 if it exists in final_seg : these segments will be passed to the clustering step
            0 otherwise
            Note that segment 0 (that covers NODATA) is also marked as invalid,
            because we cannot use it in clustering step
            """
            res_seg = eoscale_manager.get_array(future_seg[0])[0]

            size_result = np.max(res_seg) + 1

            start_valid = time.time()
            # use Cython to optimize mask computation
            ts_stats = ts.PyStats()

            start_valid = time.time()
            mask_valid_indices = ts_stats.compute_mask_valid_indices(
                res_seg, size_result
            )
            end_valid = time.time()
            logger.debug(
                f"Compute mask of valid indices (CYTHON) in "
                f"{utils.convert_time(end_valid-start_valid)}"
            )

            # Stats calculation
            stats = process_stats(
                args,
                eoscale_manager,
                future_seg,
                key_ndvi,
                key_ndwi,
                key_texture,
                size_result,
                mask_valid_indices,
            )

            time_stats = time.time()

            # Clustering #
            pred_veg, sorted_ndvi_centroids = clustering_vegetation(
                vars(args), size_result, stats[0], mask_valid_indices
            )

            logger.debug(
                f"NDVI of 1st vegetation cluster {sorted_ndvi_centroids[-args.nb_clusters_veg]=}"
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

            # Sum the two clusterings
            #     0    10   20 +
            #  0/ 1         3
            # --> 0 / 11, 13 / 21, 23
            clusters = clusters_veg + clusters_low_high_veg
            time_cluster = time.time()

            # final tab
            final_clusters = np.zeros(size_result)
            final_clusters[np.where(mask_valid_indices)] = clusters
            final_clusters[np.where(mask_valid_indices == 0)] = (
                0  # TODO : -1 or 0 ? it will be masked by valid_stack at the end
            )

            # Finalize mask #
            final_seg = eoexe.n_images_to_m_images_filter(
                inputs=[future_seg[0], key_valid_stack],
                image_filter=finalize_task,
                filter_parameters={"data": final_clusters},
                generate_output_profiles=eo_utils.single_uint8_profile,
                stable_margin=0,
                context_manager=eoscale_manager,
                multiproc_context=args.multiproc_context,
                filter_desc="Finalize processing...",
            )

            if args.save_mode == "debug":
                # Save intermediate masks
                eoscale_manager.write(
                    key=final_seg[0],
                    img_path=args.vegetationmask.replace(
                        ".tif", "_before_clean.tif"
                    ),
                )

                final_clusters[np.where(mask_valid_indices)] = pred_veg
                # Save vegetation clusters
                vegetation_clustering = eoexe.n_images_to_m_images_filter(
                    inputs=[future_seg[0], key_valid_stack],
                    image_filter=finalize_task,
                    filter_parameters={"data": final_clusters},
                    generate_output_profiles=eo_utils.single_uint8_profile,
                    stable_margin=0,
                    context_manager=eoscale_manager,
                    multiproc_context=args.multiproc_context,
                    filter_desc="Finalize processing...",
                )
                eoscale_manager.write(
                    key=vegetation_clustering[0],
                    img_path=args.vegetationmask.replace(
                        ".tif", "_vegclusters.tif"
                    ),
                )

                # Save texture clusters
                texture_clustering = eoexe.n_images_to_m_images_filter(
                    inputs=[future_seg[0], key_valid_stack],
                    image_filter=finalize_task,
                    filter_parameters={"data": pred_texture},
                    generate_output_profiles=eo_utils.single_uint8_profile,
                    stable_margin=0,
                    context_manager=eoscale_manager,
                    multiproc_context=args.multiproc_context,
                    filter_desc="Finalize processing...",
                )
                eoscale_manager.write(
                    key=texture_clustering[0],
                    img_path=args.vegetationmask.replace(
                        ".tif", "_textureclusters.tif"
                    ),
                )

            time_final = time.time()

            # Post-process : delete small holes / objects, dilate low veg areas a little bit
            # and filter output mask with the NDVI of the fist vegetation cluster
            vars(args)["min_ndvi_veg"] = sorted_ndvi_centroids[
                -args.nb_clusters_veg
            ]
            final_seg = postprocess(
                args, eoscale_manager, final_seg, key_valid_stack, key_ndvi
            )
            time_closing = time.time()

            # Write output mask #

            eoscale_manager.write(
                key=final_seg[0], img_path=args.vegetationmask
            )
            end_time = time.time()

            display_infos(
                args,
                end_time,
                t0,
                time_closing,
                time_cluster,
                time_final,
                time_seg,
                time_stack,
                time_stats,
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
    Main function to run the vegetation mask computation.
    It parses the command line arguments and calls the slurp_vegetationmask function.
    """
    args = getarguments()
    slurp_vegetationmask(**args)


if __name__ == "__main__":
    main()
