#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Compute vegetation mask of PHR image."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import traceback

from math import sqrt, ceil
from os.path import splitext
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.morphology import binary_dilation, remove_small_objects, disk, remove_small_holes

from slurp.tools import io_utils, utils
from slurp.tools import eoscale_utils as eo_utils
import eoscale.manager as eom
import eoscale.eo_executors as eoexe

# Cython module to compute stats
import stats as ts

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


def display_clusters(pdf, first_field, second_field, nb_first_group, nb_second_group, filename):
    serie1 = pdf.sort_values(by=first_field)[first_field]
    serie2 = pdf.sort_values(by=first_field)[second_field]
    plt.plot(serie1[0:nb_first_group], serie2[0:nb_first_group], '*')
    plt.plot(serie1[nb_first_group:nb_second_group], serie2[nb_first_group:nb_second_group], 'o')
    plt.plot(serie1[nb_second_group:9], serie2[nb_second_group:9], '+')
    plt.title("Clusters in three groups (" + str(second_field) + " " + str(first_field) + ")")
    plt.savefig(filename)
    plt.close()


# Segmentation #

def compute_segmentation(params: dict, img: np.ndarray, ndvi: np.ndarray) -> np.ndarray:
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
        channel_axis=None
    )

    return res_seg


def segmentation_task(input_buffers: list, input_profiles: list, params: dict) -> np.ndarray:
    """
    Segmentation

    :param list input_buffers: [im_vhr, valid_stack]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: segments
    """
    # Note : input_buffers[x][input_buffers[2]] applies the valid mask on input_buffers[x]
    # Warning : input_buffers[0] : the mask is not applied ! But we only use NDVI mode (see compute_segmentation)
    segments = compute_segmentation(params, input_buffers[0], input_buffers[1])

    # minimum segment is 1, attribute 0 to no_data pixel
    segments[np.logical_not(input_buffers[2])] = 0

    return segments


def concat_seg(previousResult, outputAlgoComputer, tile):
    """
    Concatenates SLIC segmentation in a single segmentation
    """
    # Computes max of previous result and adds this value to the current result :
    # prevents from computing a map with several identical labels !!
    num_seg = np.max(previousResult[0])

    previousResult[0][:, tile.start_y:tile.end_y+1, tile.start_x:tile.end_x+1] = outputAlgoComputer[0][:, :, :] + num_seg

    previousResult[0][:, tile.start_y:tile.end_y+1, tile.start_x:tile.end_x+1] = np.where(
        outputAlgoComputer[0][:, :, :] == 0, 0, outputAlgoComputer[0][:, :, :] + num_seg
    )


# Stats #

def compute_stats_image(input_buffer: list, input_profiles: list, params: dict) -> list:
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
        np.array(input_buffer[1:nb_primitives + 1])[:, 0, :, :],
        input_buffer[0],
        params["nb_lab"]
    )

    return [accumulator, counter]


def stats_concatenate(output_scalars, chunk_output_scalars, tile):
    # output_scalars[0] : sums of each segment
    output_scalars[0] += chunk_output_scalars[0]
    # output_scalars[1] : counter of each segment (nb pixels/segment)
    output_scalars[1] += chunk_output_scalars[1]


# Clustering #

def apply_clustering(params: dict, nb_polys: int, stats: np.ndarray) -> np.ndarray:
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
        print(f"K-Means on radiometric indices ({nb_polys} elements")

    kmeans_rad_indices = KMeans(n_clusters=9, init="k-means++", n_init=5, verbose=0, random_state=712)
    pred_veg = kmeans_rad_indices.fit_predict(
        np.stack(
            (stats[0:nb_polys], stats[nb_polys: 2 * nb_polys]), axis=1
        )
    )
    if params["debug"]:
        print(f"{np.sort(kmeans_rad_indices.cluster_centers_,axis=0)=}")

    list_clusters = pd.DataFrame.from_records(kmeans_rad_indices.cluster_centers_, columns=['ndvi', 'ndwi'])
    list_clusters_by_ndvi = list_clusters.sort_values(by='ndvi', ascending=True).index

    map_centroid = []

    nb_clusters_no_veg = 0
    nb_clusters_veg = 0
    if params["min_ndvi_veg"]:
        # Attribute veg class by threshold
        for t in range(kmeans_rad_indices.n_clusters):
            if list_clusters.iloc[t]['ndvi'] > float(params["min_ndvi_veg"]):
                map_centroid.append(VEG_CODE)
                nb_clusters_veg += 1
            elif list_clusters.iloc[t]['ndvi'] < float(params["max_ndvi_noveg"]):
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
            nb_clusters_no_veg = 9 - params["nb_clusters_veg"]
            nb_clusters_veg = params["nb_clusters_veg"]

        for t in range(kmeans_rad_indices.n_clusters):
            if t in list_clusters_by_ndvi[:nb_clusters_no_veg]:
                if params["non_veg_clusters"]:
                    l_ndvi = list(list_clusters_by_ndvi)
                    v = l_ndvi.index(t)
                    map_centroid.append(v)
                else:
                    map_centroid.append(NO_VEG_CODE)  # 0
            elif t in list_clusters_by_ndvi[nb_clusters_no_veg: 9 - params["nb_clusters_veg"]]:
                map_centroid.append(UNDEFINED_VEG)  # 10
            else:
                map_centroid.append(VEG_CODE)  # 20

    clustering = apply_map(pred_veg, map_centroid)

    figure_name = splitext(params["vegetationmask"])[0] + "_centroids_veg.png"
    if params["save_mode"] == "debug":
        display_clusters(list_clusters, "ndvi", "ndwi", nb_clusters_no_veg, (9 - nb_clusters_veg), figure_name)

    # Analysis texture
    if params["texture_mode"] != "no":
        mean_texture = stats[2 * nb_polys:]
        texture_values = np.nan_to_num(mean_texture[np.where(clustering >= UNDEFINED_VEG)])
        threshold_max = np.percentile(texture_values, params["filter_texture"])
        if params["debug"]:
            print("threshold_texture_max", threshold_max)

        # Save histograms
        if params["texture_mode"] == "debug" or params["save_mode"] == "debug":
            values, bins, _ = plt.hist(texture_values, bins=75)
            plt.clf()
            bins_center = (bins[:-1] + bins[1:]) / 2
            plt.plot(bins_center, values, color="blue")
            plt.savefig(splitext(params["vegetationmask"])[0] + "_histogram_texture.png")
            plt.close()
            index_max = np.argmax(bins_center > threshold_max) + 1
            plt.plot(bins_center[:index_max], values[:index_max], color="blue")
            plt.savefig(splitext(params["vegetationmask"])[0] + "_histogram_texture_cut" + str(
                params["filter_texture"]) + ".png")
            plt.close()

        # Clustering
        data_textures = np.transpose(texture_values)
        data_textures[data_textures > threshold_max] = threshold_max
        if params["debug"]:
            print("K-Means on texture : " + str(len(data_textures)) + " elements")

        kmeans_texture = KMeans(n_clusters=9, init="k-means++", n_init=5, verbose=0, random_state=712)
        pred_texture = kmeans_texture.fit_predict(data_textures.reshape(-1, 1))

        if params["debug"]:
            print(f"{np.sort(kmeans_texture.cluster_centers_,axis=0)=}")

        list_clusters = pd.DataFrame.from_records(kmeans_texture.cluster_centers_, columns=['mean_texture'])
        list_clusters_by_texture = list_clusters.sort_values(by='mean_texture', ascending=True).index

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
            if params["max_low_veg"]:
                # Distinction veg class by threshold
                params["nb_clusters_low_veg"] = int(
                    list_clusters[list_clusters['mean_texture'] < params["max_low_veg"]].count()
                )
            if params["nb_clusters_low_veg"] >= 7:
                nb_clusters_high_veg = 9 - params["nb_clusters_low_veg"]
            for t in range(kmeans_texture.n_clusters):
                if t in list_clusters_by_texture[:params["nb_clusters_low_veg"]]:
                    map_centroid.append(LOW_TEXTURE_CODE)
                elif t in list_clusters_by_texture[9 - nb_clusters_high_veg:]:
                    map_centroid.append(HIGH_TEXTURE_CODE)
                else:
                    map_centroid.append(MIDDLE_TEXTURE_CODE)

            figure_name = splitext(params["vegetationmask"])[0] + "_centroids_texture.png"
            if params["save_mode"] == "debug":
                if params["texture_mode"] == "debug":
                    display_clusters(list_clusters, "mean_texture", "mean_texture", 0, 9, figure_name)
                else:
                    display_clusters(list_clusters, "mean_texture", "mean_texture", params["nb_clusters_low_veg"],
                                     (9 - nb_clusters_high_veg), figure_name)

        textures = np.zeros(nb_polys)
        textures[np.where(clustering >= UNDEFINED_VEG)] = apply_map(pred_texture, map_centroid)

        # Ex : 10 (undefined) + 3 (textured) -> 13
        clustering = clustering + textures

    return clustering


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
    final_mask[np.logical_not(input_buffers[1][0])] = 255

    return final_mask


def clean_task(input_buffers: list, input_profiles: list, params: dict) -> np.ndarray:
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
        high_veg_binary = remove_small_holes(
            high_veg_binary.astype(bool), params["remove_small_objects"], connectivity=2
        ).astype(np.uint8)
        im_classif[np.logical_and(im_classif == LOW_VEG_CLASS, high_veg_binary == 1)] = UNDEFINED_TEXTURE_CLASS

    low_veg_binary = np.where(im_classif == LOW_VEG_CLASS, True, False)

    if params["remove_small_holes"]:
        low_veg_binary = remove_small_holes(
            low_veg_binary.astype(bool), params["remove_small_holes"], connectivity=2
        ).astype(np.uint8)
        im_classif[np.logical_and(im_classif > LOW_VEG_CLASS, low_veg_binary == 1)] = LOW_VEG_CLASS

    if params["binary_dilation"]:
        low_veg_binary = binary_dilation(
            low_veg_binary, disk(params["binary_dilation"])
        ).astype(np.uint8)
        im_classif[np.logical_and(im_classif > LOW_VEG_CLASS, low_veg_binary == 1)] = LOW_VEG_CLASS

    return im_classif

# MAIN #


def getarguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Compute Vegetation Mask.")

    parser.add_argument("main_config", help="First JSON file, load basis arguments")
    parser.add_argument("-user_config", help="Second JSON file, overload basis arguments if keys are the same")
    parser.add_argument("-file_vhr", help="input image (reflectances TOA)")
    parser.add_argument("-vegetationmask", help="Output classification filename")

    # primitives and texture arguments
    parser.add_argument("-valid", action="store", dest="valid_stack", help="Validity mask")
    parser.add_argument("-ndvi", action="store", dest="file_ndvi", help="NDVI filename")
    parser.add_argument("-ndwi", action="store", dest="file_ndwi", help="NDWI filename")
    parser.add_argument("-texture", action="store", dest="file_texture", help="Texture filename")
    parser.add_argument("-texture_mode", "--texture_mode", choices=["yes", "no", "debug"], action="store",
                        help="Labelize vegetation with (yes) or without (no) distinction low/high, "
                             "or get all 9 vegetation clusters without distinction low/high (debug)")
    parser.add_argument("-filter_texture", "--filter_texture", type=int,
                        help="Percentile for texture (between 1 and 99)")
    parser.add_argument("-save", choices=["none", "debug"], action="store", dest="save_mode",
                        help="Save all files (debug) or only output mask (none)")

    # segmentation arguments
    parser.add_argument("-slic_seg_size", "--slic_seg_size", type=int, help="Approximative segment size (100 by default)")
    parser.add_argument("-slic_compactness", "--slic_compactness", type=float,
                        help="Balance between color and space proximity (see skimage.slic documentation) - 0.1 by default")

    # clustering arguments
    parser.add_argument("-nbclusters", "--nb_clusters_veg", type=int,
                        help="Nb of clusters considered as vegetation (1-9), default : 3")
    parser.add_argument("-min_ndvi_veg", "--min_ndvi_veg", type=int,
                        help="Minimal mean NDVI value to consider a cluster as vegetation (overload nb clusters choice)")
    parser.add_argument("-max_ndvi_noveg", "--max_ndvi_noveg", type=int,
                        help="Maximal mean NDVI value to consider a cluster as non-vegetation (overload nb clusters choice)")
    parser.add_argument("-non_veg_clusters", "--non_veg_clusters", action="store_true",
                        help="Labelize each 'non vegetation cluster' as 0, 1, 2 (..) instead of single label (0)")
    parser.add_argument("-nbclusters_low", "--nb_clusters_low_veg", type=int,
                        help="Nb of clusters considered as low vegetation (1-9), default : 3")
    parser.add_argument("-max_low_veg", "--max_low_veg", type=int,
                        help="Maximal texture value to consider a cluster as low vegetation (overload nb clusters choice)")

    # post-processing arguments
    parser.add_argument("-binary_dilation", "--binary_dilation", type=int, action="store",
                        help="Size of square structuring element")
    parser.add_argument("-remove_small_objects", "--remove_small_objects", type=int, action="store",
                        help="The maximum area, in pixels, of a contiguous object that will be removed")
    parser.add_argument("-remove_small_holes", "--remove_small_holes", type=int, action="store",
                        help="The maximum area, in pixels, of a contiguous hole that will be filled")

    # multiprocessing arguments
    parser.add_argument("-n_workers", type=int,
                        help="Number of workers for multiprocessed tasks (primitives+segmentation)")

    # Debug argument
    parser.add_argument('--debug', action='store_true', help='Debug flag')

    return parser.parse_args()


def main():
    argparse_dict = vars(getarguments())

    # Read the JSON files
    keys = ['input', 'aux_layers', 'masks', 'ressources', 'vegetation']
    argsdict = io_utils.read_json(argparse_dict["main_config"], keys, argparse_dict.get("user_config"))

    # Overload with manually passed arguments if not None
    for key in argparse_dict.keys():
        if argparse_dict[key] is not None:
            argsdict[key] = argparse_dict[key]

    print("JSON data loaded:")
    print(argsdict)
    args = argparse.Namespace(**argsdict)

    with eom.EOContextManager(nb_workers=args.n_workers, tile_mode=True) as eoscale_manager:

        try:

            t0 = time.time()

            # Build stack with all layers #

            # Image PHR
            key_phr = eoscale_manager.open_raster(raster_path=args.file_vhr)
            args.nodata_phr = eoscale_manager.get_profile(key_phr)["nodata"]

            # Valid stack
            key_valid_stack = eoscale_manager.open_raster(raster_path=args.valid_stack)

            # NDXI
            key_ndvi = eoscale_manager.open_raster(raster_path=args.file_ndvi)
            key_ndwi = eoscale_manager.open_raster(raster_path=args.file_ndwi)

            # Texture file
            key_texture = eoscale_manager.open_raster(raster_path=args.file_texture)

            time_stack = time.time()

            # Segmentation #

            future_seg = eoexe.n_images_to_m_images_filter(inputs=[key_phr, key_ndvi, key_valid_stack],
                                                           image_filter=segmentation_task,
                                                           filter_parameters=vars(args),
                                                           generate_output_profiles=eo_utils.single_int32_profile,
                                                           stable_margin=0,
                                                           context_manager=eoscale_manager,
                                                           concatenate_filter=concat_seg,
                                                           multiproc_context="fork",
                                                           filter_desc="Segmentation processing...")

            if args.save_mode == "all" or args.save_mode == "debug":
                eoscale_manager.write(key=future_seg[0], img_path=args.vegetationmask.replace(".tif", "_slic.tif"))

            time_seg = time.time()

            # Stats #

            # Recover number total of segments
            nb_polys = np.max(eoscale_manager.get_array(future_seg[0])[0])
            if args.debug:
                print("Number of different segments detected : " + str(nb_polys))

            # Stats calculation
            params_stats = {"nb_lab": nb_polys}
            stats = eoexe.n_images_to_m_scalars(inputs=[future_seg[0], key_ndvi, key_ndwi, key_texture],
                                                image_filter=compute_stats_image,
                                                filter_parameters=params_stats,
                                                nb_output_scalars=nb_polys,
                                                context_manager=eoscale_manager,
                                                concatenate_filter=stats_concatenate,
                                                multiproc_context="fork",
                                                filter_desc="Stats ")

            # stats[0] : sum of each primitive [ <- NDVI -><- NDWI -><- texture -> ]
            # stats[1] : nb pixels by segment   [ counter  ]
            # Once the sum of each primitive is computed, we compute the mean by dividing by the size of each segment
            np.seterr(divide="ignore", invalid="ignore")

            stats[0][:nb_polys] = stats[0][:nb_polys] / stats[1][:nb_polys]
            stats[0][nb_polys: 2 * nb_polys] = stats[0][nb_polys: 2 * nb_polys] / stats[1][:nb_polys]
            stats[0][2 * nb_polys: 3 * nb_polys] = stats[0][2 * nb_polys: 3 * nb_polys] / stats[1][:nb_polys]

            # Replace NaN by 0. After clustering, NO_DATA values will be masked
            stats[0] = np.where(np.isnan(stats[0]), 0, stats[0])

            time_stats = time.time()

            # Clustering #

            clusters = apply_clustering(vars(args), nb_polys, stats[0])
            time_cluster = time.time()

            # Finalize mask #

            final_seg = eoexe.n_images_to_m_images_filter(inputs=[future_seg[0], key_valid_stack],
                                                          image_filter=finalize_task,
                                                          filter_parameters={"data": clusters},
                                                          generate_output_profiles=eo_utils.single_uint8_profile,
                                                          stable_margin=0,
                                                          context_manager=eoscale_manager,
                                                          multiproc_context="fork",
                                                          filter_desc="Finalize processing (Cython)...")

            if args.save_mode == "debug":
                eoscale_manager.write(key=final_seg[0],
                                      img_path=args.vegetationmask.replace(".tif", "_before_clean.tif"))

            time_final = time.time()

            # Closing #

            if args.texture_mode == "yes" and (
                    args.binary_dilation or args.remove_small_objects or args.remove_small_holes):
                margin = max(
                    2 * args.binary_dilation, ceil(sqrt(args.remove_small_objects)), ceil(sqrt(args.remove_small_holes))
                )
                final_seg = eoexe.n_images_to_m_images_filter(inputs=[final_seg[0], key_valid_stack],
                                                              image_filter=clean_task,
                                                              filter_parameters=vars(args),
                                                              generate_output_profiles=eo_utils.single_uint8_profile,
                                                              stable_margin=margin,
                                                              context_manager=eoscale_manager,
                                                              multiproc_context="fork",
                                                              filter_desc="Post-processing...")
            time_closing = time.time()

            # Write output mask #

            eoscale_manager.write(key=final_seg[0], img_path=args.vegetationmask)
            end_time = time.time()

            print(f"**** Vegetation mask for {args.file_vhr} (saved as {args.vegetationmask}) ****")
            print("Total time (user)       :\t" + utils.convert_time(end_time - t0))
            print("- Build_stack           :\t" + utils.convert_time(time_stack - t0))
            print("- Segmentation          :\t" + utils.convert_time(time_seg - time_stack))
            print("- Stats                 :\t" + utils.convert_time(time_stats - time_seg))
            print("- Clustering            :\t" + utils.convert_time(time_cluster - time_stats))
            print("- Finalize Cython       :\t" + utils.convert_time(time_final - time_cluster))
            print("- Post-processing       :\t" + utils.convert_time(time_closing - time_final))
            print("- Write final image     :\t" + utils.convert_time(end_time - time_closing))
            print("***")

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
