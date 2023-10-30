#!/usr/bin/python
import sys
from os.path import dirname, join, splitext
import argparse
import geopandas as gpd
import fiona
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import rasterio
from rasterio import features
import matplotlib.pyplot as plt
import time
from skimage.segmentation import slic
from skimage.segmentation import felzenszwalb
import concurrent.futures
from shapely.geometry import shape
import scipy
from slum.tools import io_utils

import eoscale.manager as eom
import eoscale.eo_executors as eoexe

# Cython module to compute stats
import stats as ts

import time

NO_VEG_CODE = 0
WATER_CODE = 3

LOW_VEG_CODE = 1
UNDEFINED_TEXTURE = 2
HIGH_VEG_CODE = 3

UNDEFINED_VEG = 10
VEG_CODE = 20

########### MISCELLANEOUS FUNCTIONS ##############


def get_transform(transform, beginx, beginy):
    """Get transform of a part of an image knowing the transform of the full image."""
    new_transform = rasterio.Affine(
        transform[0],
        transform[1],
        transform[2] + beginy * transform[0],
        transform[3],
        transform[4],
        transform[5] + beginx * transform[4]
    )
    return new_transform


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
    
    
def single_float_profile(input_profiles: list, map_params):
    profile = input_profiles[0]
    profile['count']=1
    profile['dtype']=np.float32
    profile["compress"] = "lzw"
    
    return profile

def single_int32_profile(input_profiles: list, map_params):
    profile= input_profiles[0]
    profile["count"]= 1
    profile["dtype"]= np.int32
    profile["compress"] = "lzw"
    
    return profile

def multiple_int32_profile(input_profiles: list, map_params):
    profile= input_profiles[0]
    profile["count"]= 3
    profile["dtype"]= np.int32
    profile["compress"] = "lzw"
    
    return profile

def single_uint8_profile(input_profiles: list, map_params):
    profile = input_profiles[0]
    profile["count"]= 1
    profile["dtype"]= np.uint8
    profile["compress"] = "lzw"
    
    return profile




def print_dataset_infos(dataset, prefix=""):
    """Print information about rasterio dataset."""
    print()
    print(prefix, "Image name :", dataset.name)
    print(prefix, "Image size :", dataset.width, "x", dataset.height)
    print(prefix, "Image bands :", dataset.count)
    print(prefix, "Image types :", dataset.dtypes)
    print(prefix, "Image nodata :", dataset.nodatavals, dataset.nodata)
    print(prefix, "Image crs :", dataset.crs)
    print(prefix, "Image bounds :", dataset.bounds)
    print()



        

########### Radiometric indices ##############

def compute_ndvi(input_buffers: list, 
                  input_profiles: list, 
                  params: dict) -> np.ndarray :
    """Compute Normalize Difference X Index.
    Rescale to [-1000, 1000] int16 with nodata value = 32767
    1000 * (im_b1 - im_b2) / (im_b1 + im_b2)
    """

    np.seterr(divide="ignore", invalid="ignore")

    im_ndvi = 1000.0 - (2000.0 * np.float32(input_buffers[0][params.red_band-1])) / (
        np.float32(input_buffers[0][params.nir_band-1]) + np.float32(input_buffers[0][params.red_band-1]))
    im_ndvi[np.logical_or(im_ndvi < -1000.0, im_ndvi > 1000.0)] = np.nan
    np.nan_to_num(im_ndvi, copy=False, nan=32767)
    im_ndvi = np.int16(im_ndvi)


    return im_ndvi


def compute_ndwi(input_buffers: list, 
                  input_profiles: list, 
                  params: dict) -> np.ndarray :
    """Compute Normalize Difference X Index.
    Rescale to [-1000, 1000] int16 with nodata value = 32767
    1000 * (im_b1 - im_b2) / (im_b1 + im_b2)
    """

    np.seterr(divide="ignore", invalid="ignore")

    im_ndwi = 1000.0 - (2000.0 * np.float32(input_buffers[0][params.nir_band-1])) / (
        np.float32(input_buffers[0][params.green_band-1]) + np.float32(input_buffers[0][params.nir_band-1]))
    im_ndwi[np.logical_or(im_ndwi < -1000.0, im_ndwi > 1000.0)] = np.nan
    np.nan_to_num(im_ndwi, copy=False, nan=32767)
    im_ndwi = np.int16(im_ndwi)

    return im_ndwi

########### Texture indices ##############

def texture_task(input_buffers: list, 
                  input_profiles: list, 
                  args: dict) -> np.ndarray :
    # input_buffers = [input_img]
    # Compute textures
    texture, t_texture = std_convoluted(input_buffers[0][args.nir_band - 1].astype(float), args.texture_rad, args.filter_texture, args.min_value, args.max_value)
    
    return texture
    

def std_convoluted(im, N, filter_texture, min_value, max_value):
    """Calculate the std of each pixel
    Based on a convolution with a kernel of 1 (size of the kernel given)
    """
    t0 = time.time()
    im2 = im**2
    kernel = np.ones((2*N+1, 2*N+1))
    s = scipy.signal.convolve2d(im, kernel, mode="same", boundary="symm") # Local mean with convolution
    s2 = scipy.signal.convolve2d(im2, kernel, mode="same", boundary="symm") # local mean of the squared image with convolution
    ns = kernel.size * np.ones(im.shape)
    res = np.sqrt((s2 - s**2 / ns) / ns) # std calculation
    
    # Normalization
    res = res / (max_value - min_value)
    
    return res, (time.time() - t0)


########### Segmentation ##############
                   
def compute_segmentation(args, img, ndvi):
    """Compute segmentation with SLIC """
    nseg = int(img.shape[2] * img.shape[1] / args.slic_seg_size)

    # TODO : RGB segmentation mode is not working fine. Could be deleted (keep only NDVI mode)
    if args.segmentation_mode == "RGB":
        # Note : we read RGB image.
        # data = img.reshape(img.shape[1], img.shape[2], img.shape[0])[:,:,:3] convert2lab=True,
        data = img.reshape(img.shape[1], img.shape[2], img.shape[0])[:,:,:3]
        res_seg = slic(data, compactness=float(args.slic_compactness), n_segments=nseg, sigma=1,  channel_axis = 2)
        res_seg = res_seg.reshape(1,res_seg.shape[0], res_seg.shape[1])
    else:
        # Note : we read NDVI image.
        # Estimation of the max number of segments (ie : each segment is > 100 pixels)
        res_seg = slic(ndvi.astype("double"), compactness=float(args.slic_compactness), n_segments=nseg, sigma=1, channel_axis=None)        
    
    return res_seg


def segmentation_task(input_buffers: list, 
                  input_profiles: list, 
                  args: dict) -> np.ndarray :
    # input_buffers = [input_img,ndvi]                    

    # Segmentation
    segments = compute_segmentation(args,input_buffers[0], input_buffers[1])

    return segments



def concat_seg(previousResult,outputAlgoComputer, tile):
    #outputAlgoComputer= [segments]
    num_seg= np.max(previousResult[0])
    previousResult[0][:, tile.start_y: tile.end_y + 1, tile.start_x : tile.end_x + 1] = outputAlgoComputer[0][:,:,:] + (num_seg)
    
    



########### Stats ##############

def compute_stats_image(inputBuffer: list, 
                        input_profiles: list, 
                        params: dict) -> list:
    # inputBuffer : seg, NDVI, NDWI, texture
    
    ts_stats = ts.PyStats()
    nb_primitives =  len(inputBuffer)-1
    
    # inputBuffer : list of (one band, rows, cols) images
    # [:,0,:,:] -> transform in an array (3bands, rows, cols)
    accumulator, counter = ts_stats.run_stats(np.array(inputBuffer[1:nb_primitives+1])[:,0,:,:], inputBuffer[0], params["nb_lab"])
        
    # output : [ mean of each primitive ; counter (nb pixels / seg) ]
    return [accumulator, counter]



def stats_concatenate(output_scalars, chunk_output_scalars, tile):
    # single band version
    output_scalars[0] += chunk_output_scalars[0]
    output_scalars[1] += chunk_output_scalars[1]

########### Clustering ##############

def apply_clustering(args, stats, nb_polys):
    '''
    stats[0:nb_polys] -> mean NDVI
    stats[nb_polys:2*nb_polys] -> mean NDWI
    stats[2*nb_polys:] -> mean Texture
    '''
    clustering = np.zeros(nb_polys)
    
    # Note : the seed for random generator is fixed to obtain reproductible results
    print(f"K-Means on radiometric indices ({nb_polys} elements")
    kmeans_rad_indices = KMeans(n_clusters=9,
                                 init="k-means++",
                                 n_init=5,
                                 verbose=0,
                                 random_state=712)
    pred_veg = kmeans_rad_indices.fit_predict(np.stack((stats[0:nb_polys],stats[nb_polys:2*nb_polys]),axis=1))
    print(f"{kmeans_rad_indices.cluster_centers_=}")
    
    list_clusters = pd.DataFrame.from_records(kmeans_rad_indices.cluster_centers_, columns=['ndvi', 'ndwi'])
    list_clusters_by_ndvi = list_clusters.sort_values(by='ndvi', ascending=True).index

    map_centroid = []
    
    nb_clusters_no_veg = 0
    nb_clusters_veg = 0
    if args.min_ndvi_veg:
        # Attribute veg class by threshold
        for t in range(kmeans_rad_indices.n_clusters):
            if list_clusters.iloc[t]['ndvi'] > float(args.min_ndvi_veg):
                map_centroid.append(VEG_CODE)
                nb_clusters_veg += 1
            elif list_clusters.iloc[t]['ndvi'] < float(args.max_ndvi_noveg):
                if args.non_veg_clusters:
                    l_ndvi = list(list_clusters_by_ndvi)
                    v = l_ndvi.index(t)
                    map_centroid.append(v) 
                else:
                    # 0
                    map_centroid.append(NO_VEG_CODE)
                nb_clusters_no_veg += 1
            else:
                map_centroid.append(UNDEFINED_VEG)

    else:
        # Attribute class by thirds 
        nb_clusters_no_veg = int(kmeans_rad_indices.n_clusters / 3)
        if args.nb_clusters_veg >= 7:
            nb_clusters_no_veg = 9 - args.nb_clusters_veg
            nb_clusters_veg = args.nb_clusters_veg

        for t in range(kmeans_rad_indices.n_clusters):
            if t in list_clusters_by_ndvi[:nb_clusters_no_veg]:
                if args.non_veg_clusters:
                    l_ndvi = list(list_clusters_by_ndvi)
                    v = l_ndvi.index(t)
                    map_centroid.append(v) 
                else:
                    # 0
                    map_centroid.append(NO_VEG_CODE)
            elif t in list_clusters_by_ndvi[nb_clusters_no_veg:9-args.nb_clusters_veg]:
                # 10
                map_centroid.append(UNDEFINED_VEG)
            else:
                # 20
                map_centroid.append(VEG_CODE)
                
    clustering = apply_map(pred_veg, map_centroid)

    figure_name = splitext(args.file_classif)[0] + "_centroids_veg.png"
    display_clusters(list_clusters, "ndvi", "ndwi", nb_clusters_no_veg, (9-nb_clusters_veg), figure_name)
    
    ## Analysis texture
    if not args.no_texture:
        mean_texture = 1000 * stats[2*nb_polys:]
        texture_values = np.nan_to_num(mean_texture[np.where(clustering >= UNDEFINED_VEG)])
        #texture_values = np.nan_to_num([gdf[gdf.pred_veg >= UNDEFINED_VEG].mean_texture.values])
        threshold_max = np.percentile(texture_values, args.filter_texture)
        print("threshold_texture_max", threshold_max)

        # Save histograms
        if args.debug:
            values, bins, _ = plt.hist(texture_values, bins=75)
            plt.clf()
            bins_center = (bins[:-1] + bins[1:]) / 2
            plt.plot(bins_center, values, color="blue")
            plt.savefig(splitext(args.file_classif)[0] + "_histogram_texture.png")   
            plt.close()    
            index_max = np.argmax(bins_center>threshold_max) + 1
            plt.plot(bins_center[:index_max], values[:index_max], color="blue")
            plt.savefig(splitext(args.file_classif)[0] + "_histogram_texture_cut" + str(args.filter_texture) + ".png")   
            plt.close()

        # Clustering
        data_textures = np.transpose(texture_values)
        data_textures[data_textures > threshold_max] = threshold_max
        
        #texture_values[texture_values > threshold_max] = threshold_max
        #data_texture = np.reshape(len(texture_values),1)
        print("K-Means on texture : "+str(len(data_textures))+" elements")
        kmeans_texture = KMeans(n_clusters=9,
                                init="k-means++",
                                n_init=5,
                                verbose=0,
                                random_state=712)
        pred_texture = kmeans_texture.fit_predict(data_textures.reshape(-1,1))
        print(kmeans_texture.cluster_centers_)

        list_clusters = pd.DataFrame.from_records(kmeans_texture.cluster_centers_, columns=['mean_texture'])
        list_clusters_by_texture = list_clusters.sort_values(by='mean_texture', ascending=True).index

        # Attribute class
        map_centroid = []
        nb_clusters_high_veg = int(kmeans_texture.n_clusters / 3)
        if args.max_low_veg:
            # Distinction veg class by threshold
            args.nb_clusters_low_veg = int(list_clusters[list_clusters['mean_texture'] < args.max_low_veg].count())
        if args.nb_clusters_low_veg >= 7:
            nb_clusters_high_veg = 9 - args.nb_clusters_low_veg
        for t in range(kmeans_texture.n_clusters):
            if t in list_clusters_by_texture[:args.nb_clusters_low_veg]:
                map_centroid.append(LOW_VEG_CODE)
            elif t in list_clusters_by_texture[9-nb_clusters_high_veg:]:
                map_centroid.append(HIGH_VEG_CODE)
            else:
                map_centroid.append(UNDEFINED_TEXTURE)
        '''
        if args.debug:
            # Get all clusters
            map_centroid_debug = []
            list_clusters_by_texture = list_clusters_by_texture.tolist()
            for t in range(kmeans_texture.n_clusters):
                map_centroid_debug.append(list_clusters_by_texture.index(t)) 
            gdf["Texture_debug"] = 0
            gdf.loc[gdf.pred_veg >= UNDEFINED_VEG, "Texture_debug"] = apply_map(pred_texture, map_centroid_debug)
            gdf.loc[gdf.pred_veg == UNDEFINED_VEG, "Texture_debug"] = 0
            gdf["ClasseN_debug"] = gdf["pred_veg"] + gdf["Texture_debug"]

        '''

        figure_name = splitext(args.file_classif)[0] + "_centroids_texture.png"
        display_clusters(list_clusters, "mean_texture", "mean_texture", args.nb_clusters_low_veg,
                         (9-nb_clusters_high_veg), figure_name)
        
        
        textures = np.zeros(nb_polys)
        textures[np.where(clustering >= UNDEFINED_VEG)] = apply_map(pred_texture, map_centroid)
        #gdf.loc[gdf.pred_veg >= UNDEFINED_VEG, "Texture"] = apply_map(pred_texture, map_centroid)
    
        # Ex : 10 (undefined) + 3 (textured) -> 13
        clustering = clustering + textures
    
    return clustering
        

########### Finalize ##############

def finalize_task(input_buffers: list, 
                  input_profiles: list, 
                  args: dict):
    """ Finalize mask : for each pixels in input segmentation, return mean NDVI
    """
    clustering = args["data"]

    ts_stats = ts.PyStats()
    
    final_image = ts_stats.finalize(input_buffers[0], clustering)
        
    return final_image





                        
############## MAIN FUNCTION ###############                        
                        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("im", help="input image (reflectances TOA)")
    parser.add_argument("file_classif", help="Output classification filename")
    parser.add_argument("-debug", default=False, required=False, action="store_true", 
                        help="Debug mode")

    parser.add_argument("-red", "--red_band", type=int, nargs="?", default=1, help="Red band index")
    parser.add_argument("-green", "--green_band", type=int, nargs="?", default=2, help="Green band index")
    parser.add_argument("-nir", "--nir_band", type=int, nargs="?", default=4, help="Near Infra-Red band index")
    parser.add_argument("-ndvi", default=None, required=False, action="store", dest="file_ndvi", help="NDVI filename (computed if missing option)")
    parser.add_argument("-ndwi", default=None, required=False, action="store", dest="file_ndwi", help="NDWI filename (computed if missing option)")
    parser.add_argument("-texture_rad", "--texture_rad", type=int, default=5, help="Radius for texture (std convolution) computation")
    parser.add_argument("-texture", "--filter_texture", type=int, default=90, help="Percentile for texture (between 1 and 99)")
    parser.add_argument("-no_texture", default=False, required=False, action="store_true", 
                        help="Labelize vegetation without distinction low/high")
    parser.add_argument("-save", choices=["none", "prim", "aux", "all", "debug"], default="none", required=False, action="store", dest="save_mode", help="Save all files (debug), only primitives (prim), only shp files (aux), primitives and shp files (all) or only output mask (none)")
    
    parser.add_argument("-seg", "--segmentation_mode", choices=["RGB", "NDVI"], default="NDVI", help="Image to segment : RGB or NDVI")
    parser.add_argument("-slic_seg_size", "--slic_seg_size", type=int, default=100, help="Approximative segment size (100 by default)")
    parser.add_argument("-slic_compactness", "--slic_compactness", type=float, default=0.1, help="Balance between color and space proximity (see skimage.slic documentation) - 0.1 by default")
    
    parser.add_argument("-nbclusters", "--nb_clusters_veg", type=int, default=3, help="Nb of clusters considered as vegetation (1-9), default : 3")
    parser.add_argument("-min_ndvi_veg","--min_ndvi_veg", type=int, help="Minimal mean NDVI value to consider a cluster as vegetation (overload nb clusters choice)")
    parser.add_argument("-max_ndvi_noveg","--max_ndvi_noveg", type=int, help="Maximal mean NDVI value to consider a cluster as non-vegetation (overload nb clusters choice)")
    parser.add_argument("-non_veg_clusters","--non_veg_clusters", default=False, required=False, action="store_true", 
                        help="Labelize each 'non vegetation cluster' as 0, 1, 2 (..) instead of single label (0)")
    parser.add_argument("-nbclusters_low", "--nb_clusters_low_veg", type=int, default=3,
                        help="Nb of clusters considered as low vegetation (1-9), default : 3")
    parser.add_argument("-max_low_veg","--max_low_veg", type=int, help="Maximal texture value to consider a cluster as low vegetation (overload nb clusters choice)")

    parser.add_argument("-n_workers", "--nb_workers", type=int, default=8, help="Number of workers for multiprocessed tasks (primitives+segmentation)")

    args = parser.parse_args()
    print("DBG > arguments parsed "+str(args))
                        

    with eom.EOContextManager(nb_workers = args.nb_workers, tile_mode = True) as eoscale_manager:
        input_img = eoscale_manager.open_raster(raster_path = args.im)
        t0 = time.time()
    
        #Compute NDVI
        if args.file_ndvi == None:
            ndvi = eoexe.n_images_to_m_images_filter(inputs = [input_img],
                                                           image_filter = compute_ndvi,
                                                           filter_parameters=args,
                                                           generate_output_profiles = single_float_profile,
                                                           stable_margin= 0,
                                                           context_manager = eoscale_manager,
                                                           filter_desc= "NDVI processing...")
            if args.save_mode == "all" or args.save_mode == "prim" or args.save_mode == "debug":
                eoscale_manager.write(key = ndvi[0], img_path = args.file_classif.replace(".tif","_NDVI.tif"))
        else:
            ndvi=eoscale_manager.open_raster(raster_path =args.ndvi)
        
        t_NDVI = time.time()
        
        #Compute NDWI
        if args.file_ndwi == None:
            ndwi = eoexe.n_images_to_m_images_filter(inputs = [input_img],
                                                           image_filter = compute_ndwi,
                                                           filter_parameters=args,
                                                           generate_output_profiles = single_float_profile,
                                                           stable_margin= 0,
                                                           context_manager = eoscale_manager,
                                                           filter_desc= "NDWI processing...")         
            if args.save_mode == "all" or args.save_mode == "prim" or args.save_mode == "debug":
                eoscale_manager.write(key = ndwi[0], img_path = args.file_classif.replace(".tif","_NDWI.tif"))
        else:
            ndwi= eoscale_manager.open_raster(raster_path =args.ndwi)      
        
        t_NDWI = time.time()
        
        args.min_value = np.min(eoscale_manager.get_array(input_img)[3])
        args.max_value = np.max(eoscale_manager.get_array(input_img)[3])
        
        texture = eoexe.n_images_to_m_images_filter(inputs = [input_img],
                                                    image_filter = texture_task,
                                                    filter_parameters=args,
                                                    generate_output_profiles = single_float_profile,
                                                    stable_margin= args.filter_texture,
                                                    context_manager = eoscale_manager,
                                                    filter_desc= "Texture processing...")         
        if args.save_mode == "all" or args.save_mode == "prim" or args.save_mode == "debug":
            eoscale_manager.write(key = texture[0], img_path = args.file_classif.replace(".tif","_texture.tif"))
  
        t_texture = time.time()

        #Segmentation
        future_seg = eoexe.n_images_to_m_images_filter(inputs = [input_img,ndvi[0]],
                                                           image_filter = segmentation_task,
                                                           filter_parameters=args,
                                                           generate_output_profiles = single_int32_profile, 
                                                           stable_margin= 0,
                                                           context_manager = eoscale_manager,
                                                           concatenate_filter= concat_seg, 
                                                           filter_desc= "Segmentation processing...")
    
        if args.save_mode == "all" or args.save_mode == "prim" or args.save_mode == "debug":
            eoscale_manager.write(key = future_seg[0], img_path = args.file_classif.replace(".tif","_slic.tif"))
          
        t_seg = time.time()  
            
        # Recover number total of segments
        nb_polys= np.max(eoscale_manager.get_array(future_seg[0])[0]) #len(np.unique(eoscale_manager.get_array(future_seg[0])[0]))
        print("Number of different segments detected : "+ str(nb_polys))
            
        #Stats calculation
        t5_stats = time.time()
        
        params_stats = {"nb_lab": nb_polys }
        stats = eoexe.n_images_to_m_scalars(inputs = [future_seg[0], ndvi[0], ndwi[0], texture[0]],
                                            image_filter = compute_stats_image,
                                            filter_parameters = params_stats,
                                            nb_output_scalars = nb_polys,
                                            context_manager = eoscale_manager,
                                            concatenate_filter = stats_concatenate,
                                            filter_desc = "Stats ")

        t_stats = time.time()

        # Clustering 
        clusters = apply_clustering(args, stats[0], nb_polys)
        t_cluster = time.time()
    
        # Finalize mask
        final_seg = eoexe.n_images_to_m_images_filter(inputs = [future_seg[0]],
                                                      image_filter = finalize_task,
                                                      filter_parameters={"data":clusters},
                                                      generate_output_profiles = single_uint8_profile, 
                                                      stable_margin= 0,
                                                      context_manager = eoscale_manager,
                                                      filter_desc= "Finalize processing (Cython)...")

        t_before_write = time.time()
        eoscale_manager.write(key = final_seg[0], img_path = args.file_classif)
        t_final = time.time()
        
        print(f">>> Total time = {t_final - t0:.2f}")
        print(f">>> \tNDVI = {t_NDVI - t0:.2f}")
        print(f">>> \tNDWI = {t_NDWI - t_NDVI:.2f}")
        print(f">>> \tTexture = {t_texture - t_NDWI:.2f}")
        print(f">>> \tSegmentation = {t_seg - t_texture:.2f}")
        print(f">>> \tStats = {t_stats - t_texture:.2f}")
        print(f">>> \tClustering = {t_cluster - t_stats:.2f}")
        print(f">>> \tFinalize Cython= {t_final - t_cluster:.2f}")
        print(f">>> \tWrite final image= {t_final - t_before_write:.2f}")
        print(f">>> **********************************")
        
        
        
        
if __name__ == "__main__":
    main()
