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

import time

NO_VEG_CODE = 0
WATER_CODE = 3

LOW_VEG_CODE = 1
UNDEFINED_TEXTURE = 2
HIGH_VEG_CODE = 3

UNDEFINED_VEG = 10
VEG_CODE = 20


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

def single_uint8_profile(input_profiles: list, map_params):
    profile = input_profiles[0]
    profile["count"]= 1
    profile["dtype"]= np.uint8
    profile["compress"] = "lzw"
    
    return profile

def finalize_task(input_buffers: list, 
                  input_profiles: list, 
                  args: dict):
    """ Finalize mask : for each pixels in input segmentation, return mean NDVI
    """
    datas = args['data']
    
    final_image = np.zeros(input_buffers[0][0].shape)

    nb_rows, nb_cols = input_buffers[0][0].shape
    
    for r in range(nb_rows):
        for c in range(nb_cols):
            seg = input_buffers[0][0][r][c]
            index = seg - 1
            if (datas[2][index] < 0) :
                classe = 0
            elif (datas[2][index] < 350) :
                classe = 1
            else:
                classe = 2
            final_image[r][c] = classe

    return final_image



def concat_seg(previousResult,outputAlgoComputer, tile):
    #outputAlgoComputer= [segments]
    num_seg= np.max(previousResult[0])
    previousResult[0][:, tile.start_y: tile.end_y + 1, tile.start_x : tile.end_x + 1] = outputAlgoComputer[0][:,:,:] + (num_seg)
    
    
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


def std_convoluted(im, N, filter_texture):
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
    
    # Filter extreme values
    thresh_texture = np.percentile(res, filter_texture)
    res[res > thresh_texture] = thresh_texture
    
    return res, (time.time() - t0)


def accumulate(input_buffers: list, 
                  input_profiles: list, 
                  args: dict):
    """Stats calculation
    Get the mean of each polygon of the segmentation for NDVI, NDWI and texture bands.
    """
    # Init
    segment_index = np.unique(input_buffers[0])
    nb_polys= args["nb_polys"]
    counter = np.zeros(nb_polys)
    accumulator_ndvi = np.zeros(nb_polys)
    accumulator_ndwi = np.zeros(nb_polys)
    accumulator_texture = np.zeros(nb_polys) 
    nb_channel, nb_rows, nb_cols = input_buffers[0].shape
    datas= [[]]*5  #[segment_index, counter, ndvi_mean, ndwi_mean, texture_mean]
 
    # Parse the image and set counter and sum for each polygon
    for r in range(nb_rows):
        for c in range(nb_cols):
            value = input_buffers[0][0][r][c] 
            #print(f">>> {value}")
            index = value - 1
            counter[index] += 1
            accumulator_ndvi[index] += input_buffers[2][0][r][c]
            accumulator_ndwi[index] += input_buffers[3][0][r][c]
            accumulator_texture[index] += input_buffers[1][0][r][c]
    
    # Get means for each polygon and store it in datas
    for index in range(nb_polys):
        if counter[index] > 0 :
            accumulator_ndvi[index] /= counter[index]
            accumulator_ndwi[index] /= counter[index]
            accumulator_texture[index] /= counter[index]
    
    
    # Recombine all the datas    
    datas[0]= segment_index
    datas[1]= counter
    datas[2]= accumulator_ndvi
    datas[3]= accumulator_ndwi
    datas[4]= accumulator_texture

    return datas
    

def concat_stats(previousResult,outputAlgoComputer, tile):
    #list order : [segment_index, counter, ndvi_mean, ndwi_mean, texture_mean]
    # Do ponderated mean for the 3 last list of values on every segment
    #print(f"DBG > concat_stats : {outputAlgoComputer=}")

    for i in range(len(outputAlgoComputer[0])): 
        # seg is the Ith value in outputAlgoCompute[0] : it's index is "seg - 1" in the main array
        seg = outputAlgoComputer[0][i]
        global_index = seg - 1
        
        # nb of pixels covered by seg in the current tile
        local_counter = outputAlgoComputer[1][global_index]

        # attribute the right label
        previousResult[0][global_index] = seg
        
        for j in [2,3,4] :
            #if local_counter > 0:
            # Compute stats for pixels 
            previous_sum = previousResult[j][global_index] * previousResult[1][global_index]
            current_sum  = outputAlgoComputer[j][global_index] * local_counter
            total_counter = previousResult[1][global_index] + local_counter
            if total_counter > 0:
                previousResult[j][global_index] = (previous_sum + current_sum)/(total_counter)

        # increment counter (only after the previous loop)
        previousResult[1][global_index] += local_counter
        
                    

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


def apply_clustering(args, stats):
   
    return None
    
def texture_task(input_buffers: list, 
                  input_profiles: list, 
                  args: dict) -> np.ndarray :
    # input_buffers = [input_img]
    # Compute textures
    texture, t_texture = std_convoluted(input_buffers[0][args.nir_band - 1].astype(float), args.texture_rad, args.filter_texture)
    
    return texture
    
def segmentation_task(input_buffers: list, 
                  input_profiles: list, 
                  args: dict) -> np.ndarray :
    # input_buffers = [input_img,ndvi]                    

    # Segmentation
    segments = compute_segmentation(args,input_buffers[0], input_buffers[1])

    return segments


                        
############## MAIN FUNCTION ###############                        
                        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("im", help="input image (reflectances TOA)")
    parser.add_argument("file_classif", help="Output classification filename")

    parser.add_argument("-red", "--red_band", type=int, nargs="?", default=1, help="Red band index")
    parser.add_argument("-green", "--green_band", type=int, nargs="?", default=2, help="Green band index")
    parser.add_argument("-nir", "--nir_band", type=int, nargs="?", default=4, help="Near Infra-Red band index")
    parser.add_argument("-ndvi", default=None, required=False, action="store", dest="file_ndvi", help="NDVI filename (computed if missing option)")
    parser.add_argument("-ndwi", default=None, required=False, action="store", dest="file_ndwi", help="NDWI filename (computed if missing option)")
    parser.add_argument("-texture_rad", "--texture_rad", type=int, default=5, help="Radius for texture (std convolution) computation")
    parser.add_argument("-texture", "--filter_texture", type=int, default=98, help="Percentile for texture (between 1 and 99)")
    parser.add_argument("-save", choices=["none", "prim", "aux", "all", "debug"], default="none", required=False, action="store", dest="save_mode", help="Save all files (debug), only primitives (prim), only shp files (aux), primitives and shp files (all) or only output mask (none)")
    
    parser.add_argument("-seg", "--segmentation_mode", choices=["RGB", "NDVI"], default="NDVI", help="Image to segment : RGB or NDVI")
    parser.add_argument("-slic_seg_size", "--slic_seg_size", type=int, default=100, help="Approximative segment size (100 by default)")
    parser.add_argument("-slic_compactness", "--slic_compactness", type=float, default=0.1, help="Balance between color and space proximity (see skimage.slic documentation) - 0.1 by default")
    
    parser.add_argument("-nbclusters", "--nb_clusters_veg", type=int, default=3, help="Nb of clusters considered as vegetaiton (1-9), default : 3")
    parser.add_argument("-min_ndvi_veg","--min_ndvi_veg", type=int, help="Minimal mean NDVI value to consider a cluster as vegetation (overload nb clusters choice)")
    parser.add_argument("-max_ndvi_noveg","--max_ndvi_noveg", type=int, help="Maximal mean NDVI value to consider a cluster as non-vegetation (overload nb clusters choice)")
    parser.add_argument("-non_veg_clusters","--non_veg_clusters", default=False, required=False, action="store_true", 
                        help="Labelize each 'non vegetation cluster' as 0, 1, 2 (..) instead of single label (0)")
    
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
        
        t1_NDVI = time.time()
        
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
        
        t2_NDWI = time.time()
        
        texture = eoexe.n_images_to_m_images_filter(inputs = [input_img],
                                                    image_filter = texture_task,
                                                    filter_parameters=args,
                                                    generate_output_profiles = single_float_profile,
                                                    stable_margin= args.filter_texture,
                                                    context_manager = eoscale_manager,
                                                    filter_desc= "Texture processing...")         
        if args.save_mode == "all" or args.save_mode == "prim" or args.save_mode == "debug":
            eoscale_manager.write(key = texture[0], img_path = args.file_classif.replace(".tif","_texture.tif"))
  
        t3_texture = time.time()

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
          
        t4_seg = time.time()  
            
        # Recover number total of segments
        nb_polys= np.max(eoscale_manager.get_array(future_seg[0])[0]) #len(np.unique(eoscale_manager.get_array(future_seg[0])[0]))
        print("Number of different segments detected : "+ str(nb_polys))
            
        #Stats calculation
        stats = eoexe.n_images_to_m_scalars(inputs = [future_seg[0],texture[0],ndvi[0],ndwi[0]],    # future_seg[0]=segmentation / future_seg[1]=texture
                                            image_filter = accumulate, 
                                            filter_parameters={"nb_polys":nb_polys},
                                            nb_output_scalars = 5,   # not used
                                            output_scalars= np.zeros((5,nb_polys)),  
                                            concatenate_filter = concat_stats,
                                            context_manager = eoscale_manager,
                                            filter_desc= "Statistics calculation processing...")
      
        # Clustering 
        #clusters= apply_clustering(args,stats)s

        t5_stats = time.time()
        
        # Finalize mask
        final_seg = eoexe.n_images_to_m_images_filter(inputs = [future_seg[0]],
                                                      image_filter = finalize_task,
                                                      filter_parameters={"data":stats},
                                                      generate_output_profiles = single_uint8_profile, 
                                                      stable_margin= 0,
                                                      context_manager = eoscale_manager,
                                                      filter_desc= "Finalize processing...")

        eoscale_manager.write(key = final_seg[0], img_path = args.file_classif)
        t6_final = time.time()
        
        print(f">>> Total time = {t6_final - t0:.2f}")
        print(f">>> \tNDVI = {t1_NDVI - t0:.2f}")
        print(f">>> \tNDWI = {t2_NDWI - t1_NDVI:.2f}")
        print(f">>> \tTexture {args.texture_rad=} = {t3_texture - t2_NDWI:.2f}")
        print(f">>> \tSegmentation = {t4_seg - t3_texture:.2f}")
        print(f">>> \tStats = {t5_stats - t4_seg:.2f}")
        print(f">>> \tFinalize = {t6_final - t5_stats:.2f}")
        print(f">>> **********************************")
        
        
        
        
if __name__ == "__main__":
    main()
