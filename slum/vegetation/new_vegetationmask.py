#!/usr/bin/python
import sys
import glob
import os
from os.path import dirname, join
import logging
import argparse
import subprocess
import geopandas as gpd
import fiona
import pandas as pd
import numpy as np
import numpy.ma as ma
from sklearn.cluster import KMeans
import rasterio
from rasterio import features
from rasterio.windows import Window
import matplotlib.pyplot as plt
import time
import otbApplication as otb
from skimage.segmentation import slic
from skimage.segmentation import felzenszwalb
import concurrent.futures
from shapely.geometry import shape
from tqdm import tqdm

import scipy
from slum.tools import io_utils

NO_VEG_CODE = 0
WATER_CODE = 3

LOW_VEG_CODE = 1
UNDEFINED_TEXTURE = 2
HIGH_VEG_CODE = 3

UNDEFINED_VEG = 10
VEG_CODE = 20

# Max width/height we open at once to compute spectral threlshold
MAX_BUFFER_SIZE = 5000 


def get_transform(transform, beginx, beginy):
    new_transform = rasterio.Affine(
        transform[0],
        transform[1],
        transform[2] + beginy * transform[0],
        transform[3],
        transform[4],
        transform[5] + beginx * transform[4]
    )
    return new_transform


def compute_ndxi(im_b1, im_b2):
    """Compute Normalize Difference X Index.
    Rescale to [-1000, 1000] int16 with nodata value = 32767
    1000 * (im_b1 - im_b2) / (im_b1 + im_b2)
    """

    np.seterr(divide="ignore", invalid="ignore")

    print("Compute NDI ...", end="")
    start_time = time.time()
    im_ndxi = 1000.0 - (2000.0 * np.float32(im_b2)) / (
        np.float32(im_b1) + np.float32(im_b2)
    )
    im_ndxi[np.logical_or(im_ndxi < -1000.0, im_ndxi > 1000.0)] = np.nan
    np.nan_to_num(im_ndxi, copy=False, nan=32767)
    im_ndxi = np.int16(im_ndxi)
    print("in", time.time() - start_time, "seconds.")

    return im_ndxi

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
    im2 = im**2
    kernel = np.ones((2*N+1, 2*N+1))
    s = scipy.signal.convolve2d(im, kernel, mode="same", boundary="symm") # Local mean with convolution
    s2 = scipy.signal.convolve2d(im2, kernel, mode="same", boundary="symm") # local mean of the squared image with convolution
    ns = kernel.size * np.ones(im.shape)
    res = np.sqrt((s2 - s**2 / ns) / ns) # std calculation
    
    # Filter
    thresh_texture = np.percentile(res, filter_texture)
    res[res > thresh_texture] = thresh_texture
    
    return res


def accumulate(res_seg, ndvi, ndwi, texture, transform):
    nb_polys = np.unique(res_seg).size
    counter = np.zeros(nb_polys)
    accumulator_ndvi = np.zeros(nb_polys)
    accumulator_ndwi = np.zeros(nb_polys)
    accumulator_texture = np.zeros(nb_polys)    
    
    nb_rows, nb_cols = res_seg.shape
    for r in range(nb_rows):
        for c in range(nb_cols):
            value = res_seg[r][c] - 1
            counter[value] += 1
            accumulator_ndvi[value] += ndvi[r][c]
            accumulator_ndwi[value] += ndwi[r][c]
            accumulator_texture[value] += texture[r][c]
    
    for value in range(nb_polys):
        accumulator_ndvi[value] /= counter[value]
        accumulator_ndwi[value] /= counter[value]
        accumulator_texture[value] /= counter[value]

    geometry = nb_polys*[0]
    for geom, val in rasterio.features.shapes(res_seg.astype("int16"), transform=transform):
        geometry[int(val)-1] = shape(geom)
   
    datas = {
        "geometry": geometry,
        "count": counter,
        "mean_ndvi": accumulator_ndvi,
        "mean_ndwi": accumulator_ndwi,
        "mean_texture": accumulator_texture
    }
    return pd.DataFrame(datas)
    

def compute_segmentation(args, img, mask, ndvi, ndwi, texture, transform):
    if args.algo_seg == "slic":
        print("DBG > compute_segmentation (skimage SLIC)")
        if args.segmentation_mode == "RGB":
            # Note : we read RGB image.
            nseg = int(img.shape[2] * img.shape[1] / args.slic_seg_size)
            data = img.reshape(img.shape[1], img.shape[2], img.shape[0])[:,:,:3]
            res_seg = slic(data, compactness=float(args.slic_compactness), n_segments=nseg, sigma=1, convert2lab=True, channel_axis = 2).astype("int32")            
        else:
            # Note : we read NDVI image.
            # Estimation of the max number of segments (ie : each segment is > 100 pixels)
            nseg = int(ndvi.shape[1] * ndvi.shape[0] / args.slic_seg_size)
            res_seg = slic(ndvi.astype("double"), compactness=float(args.slic_compactness), n_segments=nseg, mask=mask, sigma=1, channel_axis=None)        
    else:
        print("DBG > compute_segmentation (skimage Felzenszwalb)")
        if args.segmentation_mode == "RGB":
            # Note : we read RGB image.
            data = img.reshape(img.shape[1], img.shape[2], img.shape[0])[:,:,:3] 
            res_seg = felzenszwalb(data, scale=float(args.felzenszwalb_scale),channel_axis=2)
        else:
            # Note : we read NDVI image.
            res_seg = felzenszwalb(ndvi.astype("double"), scale=float(args.felzenszwalb_scale))
            
    ## Save res_seg in tif in debug mode ??
    
    # Stats calculation    
    df = accumulate(res_seg, ndvi, ndwi, texture, transform)
    
    return df
    

def segmentation_task(args, im_phr, im_ndvi, im_ndwi, mask_slic, transform):                
    # Compute textures
    texture = std_convoluted(im_phr[args.nir_band - 1].astype(float), 5, args.filter_texture)    
    # Segmentation
    df = compute_segmentation(args, im_phr, mask_slic, im_ndvi, im_ndwi, texture, transform)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("im", help="input image (reflectances TOA)")
    parser.add_argument("file_classif", help="Output classification filename")

    parser.add_argument("-red", "--red_band", type=int, nargs="?", default=1, help="Red band index")
    parser.add_argument("-green", "--green_band", type=int, nargs="?", default=2, help="Green band index")
    parser.add_argument("-nir", "--nir_band", type=int, nargs="?", default=4, help="Near Infra-Red band index")
    parser.add_argument("-ndvi", default=None, required=False, action="store", dest="file_ndvi", help="NDVI filename (computed if missing option)")
    parser.add_argument("-ndwi", default=None, required=False, action="store", dest="file_ndwi", help="NDWI filename (computed if missing option)")
    parser.add_argument("-save", choices=["none", "prim", "aux", "all", "debug"], default="none", required=False, action="store", dest="save_mode", help="Save all files (debug), only primitives (prim), only shp files (aux), primitives and shp files (all) or only output mask (none)")
    
    parser.add_argument("-seg", "--segmentation_mode", choices=["RGB", "NDVI"], default="NDVI", help="Image to segment : RGB or NDVI")
    parser.add_argument("-algo_seg", "--algo_seg", choices=["slic", "felz"], default="slic", required=False, action="store", help="Use SkImage SLIC algorithm (slic) or SkImage Felzenszwalb algorithm (felz) for segmentation")
    parser.add_argument("-slic_seg_size", "--slic_seg_size", type=int, default=100, help="Approximative segment size (100 by default)")
    parser.add_argument("-slic_compactness", "--slic_compactness", type=float, default=0.1, help="Balance between color and space proximity (see skimage.slic documentation) - 0.1 by default")
    parser.add_argument("-felz", "--felzenszwalb", default=False, help="Use SkImage Felzenszwalb algorithm for segmentation")
    parser.add_argument("-felz_scale", "--felzenszwalb_scale", type=float, default=1.0, help="Scale parameter for Felzenszwalb algorithm")
    #parser.add_argument("-ref", "--reference_data", nargs="?",
    #                    help="Compute a confusion matrix with this reference data")
    parser.add_argument("-texture", "--filter_texture", type=int, default=98, help="Percentile for texture (between 1 and 99)")
    #parser.add_argument("-nbclusters", "--nb_clusters_veg", type=int, default=3, help="Nb of clusters considered as vegetaiton (1-9), default : 3")
    #parser.add_argument("-min_ndvi_veg","--min_ndvi_veg", type=float, help="Minimal mean NDVI value to consider a cluster as vegetation (overload nb clusters choice)")
    #parser.add_argument("-max_ndvi_noveg","--max_ndvi_noveg", type=float, help="Maximal mean NDVI value to consider a cluster as non-vegetation (overload nb clusters choice)")
    #parser.add_argument("-non_veg_clusters","--non_veg_clusters", default=False, required=False, action="store_true", 
    #                    help="Labelize each 'non vegetation cluster' as 0, 1, 2 (..) instead of single label (0)")
    ##parser.add_argument("-input_veg_centroids", "--input_veg_centroids", help="Input vegetation centroids file")
    parser.add_argument("-startx", "--startx", type=int, default=0, help="Start x coordinates (crop ROI)")
    parser.add_argument("-starty", "--starty", type=int, default=0, help="Start y coordinates (crop ROI)")
    parser.add_argument("-sizex", "--sizex", type=int, help="Size along x axis (crop ROI)")
    parser.add_argument("-sizey", "--sizey", type=int, help="Size along y axis (crop ROI)")
    parser.add_argument("-buffer", "--buffer_dimension", type=int, default=512, help="Buffer dimension")
    parser.add_argument("-n_workers", "--nb_workers", type=int, default=8, help="Number of workers for multiprocessed tasks (primitives+segmentation)")
    parser.add_argument("-mask_slic_bool", "--mask_slic_bool", default=False,
                        help="Boolean value wether to use a mask during slic calculation or not")
    parser.add_argument("-mask_slic_file", "--mask_slic_file", help="Raster mask file to use if mask_slic_bool==True")
    parser.add_argument("-no_clean", "--no_clean", default=False, help="Keep temporary files")
    args = parser.parse_args()
    print("DBG > arguments parsed "+str(args))
    
    t0 = time.time()
    
    image = args.im
    result = args.file_classif
    buffer_dimension = args.buffer_dimension
    
    ds_phr = rasterio.open(image)
    
    args.crs = ds_phr.crs
    args.transform = ds_phr.transform
    args.rpc = ds_phr.tags(ns="RPC")
    
    im_phr = ds_phr.read()
    
    # Compute NDVI
    ## Remarque : ajouter la prise en compte des nodata ??
    if args.file_ndvi:
        ds_ndvi = rasterio.open(args.file_ndvi)
        print_dataset_infos(ds_ndvi, "NDVI")
        ds_ndvi.read()  # necessary for a true clean with del
        im_ndvi = ds_ndvi.read(1)
        ds_ndvi.close()
        del ds_ndvi        
    else:        
        im_ndvi = compute_ndxi(
            im_phr[args.nir_band - 1],
            im_phr[args.red_band - 1],
        )   
        if (args.save_mode != "none" and args.save_mode != "aux"):
            io_utils.save_image(
                im_ndvi,
                join(dirname(args.file_classif), "ndvi.tif"),
                args.crs,
                args.transform,
                nodata=32767,
                rpc=args.rpc,
            )
            
    # Compute NDWI
    ## Remarque : ajouter la prise en compte les nodata ??
    if args.file_ndwi:
        ds_ndwi = rasterio.open(args.file_ndwi)
        print_dataset_infos(ds_ndwi, "NDWI")
        ds_ndwi.read()  # necessary for a true clean with del
        im_ndwi = ds_ndwi.read(1)
        ds_ndwi.close()
        del ds_ndwi        
    else:
        im_ndwi = compute_ndxi(
            im_phr[args.green_band - 1],
            im_phr[args.nir_band - 1],
        )
        if (args.save_mode != "none" and args.save_mode != "aux"):
            io_utils.save_image(
                im_ndwi,
                join(dirname(args.file_classif), "ndwi.tif"),
                args.crs,
                args.transform,
                nodata=32767,
                rpc=args.rpc,
            )

    # Segmentation
    startx = 0
    stopx = ds_phr.width
    starty = 0
    stopy = ds_phr.height

    if args.sizex:
        startx = max(args.startx, 0)
        stopx = min(startx + args.sizex, ds_phr.width)
    if args.sizey:
        starty = max(args.starty, 0)
        stopy = min(starty + args.sizey, ds_phr.height)
    
    ds_phr.close()
    
    if args.mask_slic_bool =="True":
        ds_mask = rasterio.open(args.mask_slic_file)
        print_dataset_infos(ds_mask, "SLIC MASK")
        mask_slic = ds_mask.read(1).astype(bool)
    else:
        mask_slic = None
        
    ## Parallelisation avec concurrent (to remplace with eoscale)
    cptx = 0
    cpty = 0
    future_seg = []
    list_res = []
    

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.nb_workers) as executor:
        while (startx + cptx * buffer_dimension < stopx):
            beginx = startx + cptx * buffer_dimension
            endx = min(beginx + buffer_dimension, stopx)
            print("Segmentation de "+str(args.im)+ " entre "+str(beginx)+" et "+str(endx))
            while (starty + cpty * buffer_dimension < stopy):
                beginy = starty + cpty * buffer_dimension
                endy = min(beginy + buffer_dimension, stopy)
                mask = mask_slic[beginx:endx, beginy:endy] if args.mask_slic_bool =="True" else None
                future_seg.append(executor.submit(
                    segmentation_task, 
                    args, 
                    im_phr[:, beginx:endx, beginy:endy],  
                    im_ndvi[beginx:endx, beginy:endy], 
                    im_ndwi[beginx:endx, beginy:endy], 
                    mask,
                    get_transform(args.transform, beginx, beginy)
                ))
                cpty += 1
            cptx += 1
            cpty = 0
        
        
        for seg in concurrent.futures.as_completed(future_seg):
            try:
                df_res = seg.result()
                list_res.append(df_res)
                    
            except Exception as e:
                print("Exception ---> "+str(e))
            """
            else:
                time_segmentation += t_seg
                time_primitives += t_prim
            """

    df = pd.concat(list_res, ignore_index=True)
    
    cols = df.columns.tolist()
    df["polygon_id"] = np.arange(1, len(df.index) + 1)
    df = df[["polygon_id"] + cols]      

    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=args.crs)
    if args.save_mode == "debug":
        gdf.to_file("segmentation.shp")

    print(gdf)
    
    
if __name__ == "__main__":
    main()