#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Compute vegetation mask of PHR image."""


import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import time

from math import sqrt, ceil
from os.path import splitext, dirname, join
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.morphology import binary_dilation, remove_small_objects, disk, remove_small_holes

from slurp.tools import utils, io_utils, eoscale_utils as eo_utils
from slurp.prepare import aux_files as aux
from slurp.prepare.primitives import compute_ndvi, compute_ndwi
import eoscale.manager as eom
import eoscale.eo_executors as eoexe

# Cython module to compute stats
import stats as ts

NO_VEG_CODE = 0    # Water, other non vegetated areas
UNDEFINED_VEG = 10  # Non vegetated or few vegetation (weak NDVI signal)
VEG_CODE = 20      # Vegetation

LOW_TEXTURE_CODE = 1     # Smooth areas (could be low vegetation or bare soil)
MIDDLE_TEXTURE_CODE = 2  # Middle texture areas (could be high vegetation)
HIGH_TEXTURE_CODE = 3    # High texture (could be high vegetation)

LOW_VEG_CLASS = VEG_CODE + LOW_TEXTURE_CODE
UNDEFINED_TEXTURE_CLASS = VEG_CODE + MIDDLE_TEXTURE_CODE


########### MISCELLANEOUS FUNCTIONS ##############

def apply_map(pred, map_centroids):
    return np.array(list(map(lambda n: map_centroids[n], pred)))


def display_clusters(pdf, first_field, second_field, nb_first_group, nb_second_group, filename):
    serie1 = pdf.sort_values(by=first_field)[first_field]
    serie2 = pdf.sort_values(by=first_field)[second_field]
    plt.plot(serie1[0:nb_first_group], serie2[0:nb_first_group], '*')
    plt.plot(serie1[nb_first_group:nb_second_group], serie2[nb_first_group:nb_second_group], 'o')
    plt.plot(serie1[nb_second_group:9], serie2[nb_second_group:9], '+')
    plt.title("Clusters in three groups ("+str(second_field)+" "+str(first_field)+")")
    plt.savefig(filename)
    plt.close()


########### Segmentation ##############
                   
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
    
    previousResult[0][:, tile.start_y: tile.end_y + 1, tile.start_x: tile.end_x + 1] = outputAlgoComputer[0][:, :, :] + num_seg
    
    previousResult[0][:, tile.start_y: tile.end_y + 1, tile.start_x: tile.end_x + 1] = np.where(
        outputAlgoComputer[0][:, :, :] == 0,
        0,
        outputAlgoComputer[0][:, :, :] + num_seg
    )


########### Stats ##############

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
        np.array(input_buffer[1:nb_primitives+1])[:, 0, :, :],
        input_buffer[0],
        params["nb_lab"]
    )

    return [accumulator, counter]


def stats_concatenate(output_scalars, chunk_output_scalars, tile):
    # output_scalars[0] : sums of each segment
    output_scalars[0] += chunk_output_scalars[0]
    # output_scalars[1] : counter of each segment (nb pixels/segment)
    output_scalars[1] += chunk_output_scalars[1]


########### Clustering ##############

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
        display_clusters(list_clusters, "ndvi", "ndwi", nb_clusters_no_veg, (9-nb_clusters_veg), figure_name)
    
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
            plt.savefig(splitext(params["vegetationmask"])[0] + "_histogram_texture_cut" + str(params["filter_texture"]) + ".png")   
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
                elif t in list_clusters_by_texture[9-nb_clusters_high_veg:]:
                    map_centroid.append(HIGH_TEXTURE_CODE)
                else:
                    map_centroid.append(MIDDLE_TEXTURE_CODE)
                    

        figure_name = splitext(params["vegetationmask"])[0] + "_centroids_texture.png"
        if params["save_mode"] == "debug":
            if params["texture_mode"] == "debug":
                display_clusters(list_clusters, "mean_texture", "mean_texture", 0, 9, figure_name)
            else:
                display_clusters(list_clusters, "mean_texture", "mean_texture", params["nb_clusters_low_veg"],
                                 (9-nb_clusters_high_veg), figure_name)       
        
        textures = np.zeros(nb_polys)
        textures[np.where(clustering >= UNDEFINED_VEG)] = apply_map(pred_texture, map_centroid)
    
        # Ex : 10 (undefined) + 3 (textured) -> 13
        clustering = clustering + textures
    
    return clustering
        

########### Finalize ##############

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

                        
############## MAIN FUNCTION ###############                        
                        
def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-file_vhr", help="input image (reflectances TOA)")
    parser.add_argument("-vegetationmask", help="Output classification filename")
    
    # primitives and texture arguments
    parser.add_argument("-red", "--red_band", dest="red", type=int, nargs="?", default=1, help="Red band index")
    parser.add_argument("-green", "--green_band", dest="green", type=int, nargs="?", default=2, help="Green band index")
    parser.add_argument("-nir", "--nir_band", dest="nir", type=int, nargs="?", default=4,
                        help="Near Infra-Red band index")
    parser.add_argument("-ndvi", default=None, required=False, action="store", dest="file_ndvi",
                        help="NDVI filename (computed if missing option)")
    parser.add_argument("-ndwi", default=None, required=False, action="store", dest="file_ndwi",
                        help="NDWI filename (computed if missing option)")
    parser.add_argument("-texture", default=None, required=False, action="store", dest="file_texture",
                        help="Texture filename (computed if missing option)")
    parser.add_argument("-texture_mode", "--texture_mode", choices=["yes", "no", "debug"], default="yes",
                        required=False, action="store",
                        help="Labelize vegetation with (yes) or without (no) distinction low/high, "
                             "or get all 9 vegetation clusters without distinction low/high (debug)")
    parser.add_argument("-texture_rad", "--texture_rad", type=int, default=5,
                        help="Radius for texture (std convolution) computation")
    parser.add_argument("-filter_texture", "--filter_texture", type=int, default=90,
                        help="Percentile for texture (between 1 and 99)")
    parser.add_argument("-file_cloud_gml", "--file_cloud_gml", type=str, required=False, action="store",
                        help="Cloud file in .GML format")
    parser.add_argument("-save", choices=["none", "prim", "aux", "all", "debug"], default="none", required=False,
                        action="store", dest="save_mode",
                        help="Save all files (debug), only primitives (prim), only texture and segmentation files (aux),"
                             " primitives, texture and segmentation files (all) or only output mask (none)")
    
    # segmentation arguments
    parser.add_argument("-slic_seg_size", "--slic_seg_size", type=int, default=100,
                        help="Approximative segment size (100 by default)")
    parser.add_argument("-slic_compactness", "--slic_compactness", type=float, default=0.1,
                        help="Balance between color and space proximity (see skimage.slic documentation) - 0.1 by default")
    
    # clustering arguments
    parser.add_argument("-nbclusters", "--nb_clusters_veg", type=int, default=3,
                        help="Nb of clusters considered as vegetation (1-9), default : 3")
    parser.add_argument("-min_ndvi_veg", "--min_ndvi_veg", type=int,
                        help="Minimal mean NDVI value to consider a cluster as vegetation (overload nb clusters choice)")
    parser.add_argument("-max_ndvi_noveg", "--max_ndvi_noveg", type=int,
                        help="Maximal mean NDVI value to consider a cluster as non-vegetation (overload nb clusters choice)")
    parser.add_argument("-non_veg_clusters", "--non_veg_clusters", default=False, required=False, action="store_true",
                        help="Labelize each 'non vegetation cluster' as 0, 1, 2 (..) instead of single label (0)")
    parser.add_argument("-nbclusters_low", "--nb_clusters_low_veg", type=int, default=3,
                        help="Nb of clusters considered as low vegetation (1-9), default : 3")
    parser.add_argument("-max_low_veg", "--max_low_veg", type=int,
                        help="Maximal texture value to consider a cluster as low vegetation (overload nb clusters choice)")
    
    #post-processing arguments
    parser.add_argument("-binary_dilation", "--binary_dilation", type=int, required=False, default=0, action="store",
                        help="Size of square structuring element")
    parser.add_argument("-remove_small_objects", "--remove_small_objects", type=int, required=False, default=0,
                        action="store", help="The maximum area, in pixels, of a contiguous object that will be removed")
    parser.add_argument("-remove_small_holes", "--remove_small_holes", type=int, required=False, default=0,
                        action="store", help="The maximum area, in pixels, of a contiguous hole that will be filled")
    
    # multiprocessing arguments
    parser.add_argument("-n_workers", "--nb_workers", type=int, default=8,
                        help="Number of workers for multiprocessed tasks (primitives+segmentation)")

    # Debug argument
    parser.add_argument('--debug', action='store_true', help='Debug flag')

    args = parser.parse_args(args)
    print("DBG > arguments parsed " + str(args))
                        
    ds_phr = rio.open(args.file_vhr)
    args.nodata_phr = ds_phr.nodata
    ds_phr.close()
        
    with eom.EOContextManager(nb_workers = args.nb_workers, tile_mode = True) as eoscale_manager:
        input_img = eoscale_manager.open_raster(raster_path = args.file_vhr)
        t0 = time.time()
        
        # Get cloud mask if any
        if args.file_cloud_gml:
            cloud_mask_array = np.logical_not(
                aux.cloud_from_gml(args.file_cloud_gml, args.file_phr)
            )
            #save cloud mask
            io_utils.save_image(cloud_mask_array,
                    join(dirname(args.vegetationmask), "nocloud.tif"),
                    args.crs,
                    args.transform,
                    None,
                    args.rpc,
                    tags=args.__dict__,
            )
            mask_nocloud_key = eoscale_manager.open_raster(raster_path = join(dirname(args.vegetationmask), "nocloud.tif"))   
                
        else:
            # Get profile from im_phr
            profile = eoscale_manager.get_profile(input_img)
            profile["count"] = 1
            profile["dtype"] = np.uint8
            mask_nocloud_key = eoscale_manager.create_image(profile)
            eoscale_manager.get_array(key=mask_nocloud_key).fill(1)
        
        # Global validity mask construction
        valid_stack_key = eoexe.n_images_to_m_images_filter(inputs=[input_img, mask_nocloud_key],
                                                            image_filter=utils.compute_valid_stack_clouds,
                                                            filter_parameters=vars(args),
                                                            generate_output_profiles=eo_utils.single_bool_profile,
                                                            stable_margin=0,
                                                            context_manager=eoscale_manager,
                                                            multiproc_context="fork",
                                                            filter_desc="Valid stack processing...")

        # Compute NDVI
        if args.file_ndvi is None:
            ndvi = eoexe.n_images_to_m_images_filter(inputs=[input_img, valid_stack_key[0]],
                                                     image_filter=compute_ndvi,
                                                     filter_parameters=vars(args),
                                                     generate_output_profiles=eo_utils.single_int16_profile,
                                                     stable_margin=0,
                                                     context_manager=eoscale_manager,
                                                     multiproc_context="fork",
                                                     filter_desc="NDVI processing...")
            if args.save_mode == "all" or args.save_mode == "prim" or args.save_mode == "debug":
                eoscale_manager.write(key = ndvi[0], img_path = args.vegetationmask.replace(".tif","_NDVI.tif"))
        else:
            ndvi = [eoscale_manager.open_raster(raster_path=args.file_ndvi)]
        
        t_NDVI = time.time()
        
        # Compute NDWI
        if args.file_ndwi is None:
            ndwi = eoexe.n_images_to_m_images_filter(inputs=[input_img, valid_stack_key[0]],
                                                     image_filter=compute_ndwi,
                                                     filter_parameters=vars(args),
                                                     generate_output_profiles=eo_utils.single_int16_profile,
                                                     stable_margin=0,
                                                     context_manager=eoscale_manager,
                                                     multiproc_context="fork",
                                                     filter_desc="NDWI processing...")
            if args.save_mode == "all" or args.save_mode == "prim" or args.save_mode == "debug":
                eoscale_manager.write(key = ndwi[0], img_path = args.vegetationmask.replace(".tif","_NDWI.tif"))
        else:
            ndwi = [eoscale_manager.open_raster(raster_path=args.file_ndwi)]
        
        t_NDWI = time.time()

        # Recover extrema of the input image
        args.min_value = np.min(eoscale_manager.get_array(input_img)[3])
        args.max_value = np.max(eoscale_manager.get_array(input_img)[3])
        
        # Compute texture
        if args.file_texture is None:
            texture = eoexe.n_images_to_m_images_filter(inputs=[input_img, valid_stack_key[0]],
                                                        image_filter=aux.texture_task,
                                                        filter_parameters=vars(args),
                                                        generate_output_profiles=eo_utils.single_float_profile,
                                                        stable_margin=args.texture_rad,
                                                        context_manager=eoscale_manager,
                                                        multiproc_context="fork",
                                                        filter_desc="Texture processing...")
            if args.save_mode == "all" or args.save_mode == "aux" or args.save_mode == "debug":
                eoscale_manager.write(key = texture[0], img_path = args.vegetationmask.replace(".tif","_texture.tif"))
        else:
            texture = [eoscale_manager.open_raster(raster_path=args.file_texture)]
  
        t_texture = time.time()

        # Segmentation
        future_seg = eoexe.n_images_to_m_images_filter(inputs=[input_img, ndvi[0], valid_stack_key[0]],
                                                       image_filter=segmentation_task,
                                                       filter_parameters=vars(args),
                                                       generate_output_profiles=eo_utils.single_int32_profile,
                                                       stable_margin=0,
                                                       context_manager=eoscale_manager,
                                                       concatenate_filter=concat_seg,
                                                       multiproc_context="fork",
                                                       filter_desc="Segmentation processing...")
    
        if args.save_mode == "all" or args.save_mode == "aux" or args.save_mode == "debug":
            eoscale_manager.write(key = future_seg[0], img_path = args.vegetationmask.replace(".tif","_slic.tif"))
          
        t_seg = time.time()  
            
        # Recover number total of segments
        nb_polys = np.max(eoscale_manager.get_array(future_seg[0])[0])
        if args.debug:
            print("Number of different segments detected : " + str(nb_polys))
            
        # Stats calculation
        params_stats = {"nb_lab": nb_polys}
        stats = eoexe.n_images_to_m_scalars(inputs=[future_seg[0], ndvi[0], ndwi[0], texture[0]],
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
        
        t_stats = time.time()
        
        # Clustering 
        clusters = apply_clustering(vars(args), nb_polys, stats[0])
        t_cluster = time.time()       
        
        # Finalize mask
        final_seg = eoexe.n_images_to_m_images_filter(inputs=[future_seg[0], valid_stack_key[0]],
                                                      image_filter=finalize_task,
                                                      filter_parameters={"data": clusters},
                                                      generate_output_profiles=eo_utils.single_uint8_profile,
                                                      stable_margin=0,
                                                      context_manager=eoscale_manager,
                                                      multiproc_context="fork",
                                                      filter_desc="Finalize processing (Cython)...")
        
        if args.save_mode == "debug":
            eoscale_manager.write(key = final_seg[0], img_path = args.vegetationmask.replace(".tif","_before_clean.tif"))
        
        t_final = time.time()

        # Closing
        if args.texture_mode == "yes" and (args.binary_dilation or args.remove_small_objects or args.remove_small_holes):
            margin = max(
                2 * args.binary_dilation, ceil(sqrt(args.remove_small_objects)), ceil(sqrt(args.remove_small_holes))
            )
            final_seg = eoexe.n_images_to_m_images_filter(inputs=[final_seg[0], valid_stack_key[0]],
                                                          image_filter=clean_task,
                                                          filter_parameters=vars(args),
                                                          generate_output_profiles=eo_utils.single_uint8_profile,
                                                          stable_margin=margin,
                                                          context_manager=eoscale_manager,
                                                          multiproc_context="fork",
                                                          filter_desc="Post-processing...")
        t_closing = time.time()
        
        # Write output mask
        eoscale_manager.write(key = final_seg[0], img_path = args.vegetationmask)
        t_write = time.time()

        if args.debug:
            print(f">>> Total time = {t_final - t0:.2f}")
            print(f">>> \tNDVI = {t_NDVI - t0:.2f}")
            print(f">>> \tNDWI = {t_NDWI - t_NDVI:.2f}")
            print(f">>> \tTexture = {t_texture - t_NDWI:.2f}")
            print(f">>> \tSegmentation = {t_seg - t_texture:.2f}")
            print(f">>> \tStats = {t_stats - t_seg:.2f}")
            print(f">>> \tClustering = {t_cluster - t_stats:.2f}")
            print(f">>> \tFinalize Cython = {t_final - t_cluster:.2f}")
            print(f">>> \tPost-processing (clean) = {t_closing - t_final:.2f}")
            print(f">>> \tWrite final image = {t_write - t_closing:.2f}")
            print(f">>> **********************************")
        
        
if __name__ == "__main__":
    main()
