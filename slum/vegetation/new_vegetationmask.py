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
    ones = np.ones(im.shape)
    
    kernel = np.ones((2*N+1, 2*N+1))
    s = scipy.signal.convolve2d(im, kernel, mode="same")
    s2 = scipy.signal.convolve2d(im2, kernel, mode="same")
    ns = scipy.signal.convolve2d(ones, kernel, mode="same")
    
    res = np.sqrt((s2 - s**2 / ns) / ns)
    thresh_texture = np.percentile(res, filter_texture)
    res[res > thresh_texture] = thresh_texture
    
    return res
    

def get_stats(res_slic, primitives, geom, value):
    # Functions to use
    functions = {
        'min': np.min,
        'max': np.max,
        'mean': np.mean,
        'std': np.std
    }
    # Filter primitives (select the zone of the polygon)
    mask = (res_slic == value)
    dataset = primitives[:, mask]
    
    # Generate statistics
    if dataset.shape[1] == 0:
        nb_stats = len(functions.keys())*dataset.shape[0]
        statistics = np.concatenate(([geom, 0.0], nb_stats*[None]), axis=None)
    else:
        stats = np.array([[float(function(data)) for key, function in functions.items()] for data in dataset])
        statistics = np.concatenate(([geom, float(dataset.shape[1])], stats.flatten()), axis=None)
    
    return statistics


def compute_stats(primitives, res_slic, args):
    # Stats calculation    
    geometry = [get_stats(res_slic, primitives, shape(geom), int(val)) for geom, val in rasterio.features.shapes(res_slic, transform=args.transform)]
    
    # Generate geodataframe of polygons with statistics
    columns = ["geometry", "count"]
    stats = ["min", "max", "mean", "std"]
    for i in range(len(primitives)):
        columns = columns + [stat + "_" + str(i) for stat in stats]
    df = pd.DataFrame(geometry, columns=columns)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=args.crs)
    
    #print(gdf)
    gdf.to_file("segmentation_rastertools.shp")  
    
    return gdf


def compute_segmentation(args, img, range, segmentation, primitives):
    t0 = time.time()
    segmentation_raster = segmentation.replace(".shp", ".tif")

    if args.algo_seg == "slic":
        print("DBG > compute_segmentation (skimage SLIC)")
        if args.segmentation_mode == "RGB":
            nseg = int(img.shape[2] * img.shape[1] / args.slic_seg_size)
            data = img.reshape(img.shape[1], img.shape[2], img.shape[0])[:,:,:3]
            res_slic = slic(data, compactness=float(args.slic_compactness), n_segments=nseg, sigma=1, convert2lab=True, channel_axis = 2).astype("int32")
            #save_image(res_slic.astype("int16"), segmentation_raster, crs=ds_img.crs, transform=ds_img.transform, rpc=ds_img.tags(ns='RPC'), nodata=0)              
        else:
            # Note : we read primitives. NDVI is the first band
            ndvi = primitives[1, :, :]
            # Estimation of the max number of segments (ie : each segment is > 100 pixels)
            nseg = int(img.shape[1] * img.shape[0] / args.slic_seg_size)
            if args.mask_slic_bool == "True" :
                # Open mask file with good shape
                ### Ajouter le masque en argument ###
                print("To do...")
                """
                with rasterio.open(args.mask_slic_file) as ds_mask:
                        mask=ds_mask.read(1)
                        mask = mask.astype(bool)
                        res_slic = slic(ndvi.astype("double"), compactness=float(args.slic_compactness), n_segments=nseg, mask=mask, sigma=1, channel_axis=None)
                """
            else:
                res_slic = slic(ndvi.astype("double"), compactness=float(args.slic_compactness), n_segments=nseg, sigma=1, channel_axis=None).astype("int32")
            io_utils.save_image(res_slic, segmentation_raster, args.crs, args.transform, 1, args.rpc)
            #save_image(res_slic.astype("int16"), segmentation_raster, crs=ds_img.crs, transform=ds_img.transform, rpc=ds_img.tags(ns='RPC'), nodata=1)
    """
    else:
        print("DBG > compute_segmentation (skimage Felzenszwalb)" + str(primitives))
        if args.segmentation_mode == "RGB":
            with rasterio.open(image) as ds_img :
                # Note : we read RGB image.
                img = ds_img.read()
                data = img.reshape(img.shape[1], img.shape[2], img.shape[0])[:,:,:3] 
                res_felz = felzenszwalb(data, scale=float(args.felzenszwalb_scale),channel_axis=2)
                save_image(res_felz.astype("int16"), segmentation_raster, crs=ds_img.crs, transform=ds_img.transform,
                     rpc=ds_img.tags(ns='RPC'), nodata=0)
        else:
            with rasterio.open(primitives) as ds_img :
                # Note : we read primitives. NDVI is the first band
                img = ds_img.read()
                ndvi = 1000 * img[1, :, :]
                                
                res_felz = felzenszwalb(ndvi.astype("double"), scale=float(args.felzenszwalb_scale))
                save_image(res_felz.astype("int16"), segmentation_raster, crs=ds_img.crs, transform=ds_img.transform,
                     rpc=ds_img.tags(ns='RPC'), nodata=0)
    """
    """
    # WITH OTB
    t0 = time.time()
    app_ZonalStats = otb.Registry.CreateApplication("ZonalStatistics")
    app_ZonalStats.SetParameterString("in", "primitives.tif")
    app_ZonalStats.SetParameterString("inzone", "labelimage")
    app_ZonalStats.SetParameterString("inzone.labelimage.in",segmentation_raster)
    app_ZonalStats.SetParameterString("out.vector.filename", segmentation)
    app_ZonalStats.ExecuteAndWriteOutput()
    print("ZonalStats OTB :", time.time()-t0)
    """
    
    # WITHOUT OTB
    t0 = time.time()
    gdf = compute_stats(primitives, res_slic, args)
    print("ZonalStats without OTB :", time.time()-t0)
    
    sys.exit(-1)
    

def segmentation_task(args, beginx, beginy, im_ndvi, im_ndwi):
    print("Segmentation de "+str(args.im)+ " entre "+str(beginx)+" et "+str(beginx+args.buffer_dimension))
    
    #image        = "image_"+str(beginx)+"_"+str(beginy)+".tif"
    #primitives   = "primitives_"+str(beginx)+"_"+str(beginy)+".tif"
    segmentation = "segmentation_"+str(beginx)+"_"+str(beginy)+".shp"
    
    ds_im = rasterio.open(args.im)
    crop_im = ds_im.read(window=Window(beginx, beginy, args.buffer_dimension, args.buffer_dimension))
    io_utils.save_image_n_bands(crop_im, "cropped.tif", ds_im.crs, ds_im.transform, 0, ds_im.tags(ns='RPC'))
    ds_im.close()
    
    if args.mask_slic_bool =="True":
        print("Crop of the mask between " + str(beginx) + " and " + 
              str(beginx + args.buffer_dimension) + " / " + str(beginy) + 
              " and " + str(beginy + args.buffer_dimension))
        ds_mask = rasterio.open(args.mask_slic_file)
        crop_mask = ds_mask.read(window=Window(beginx, beginy, args.buffer_dimension, args.buffer_dimension))
        
    crop_ndvi = im_ndvi[beginx:beginx+args.buffer_dimension, beginy:beginy+args.buffer_dimension]
    crop_ndwi = im_ndwi[beginx:beginx+args.buffer_dimension, beginy:beginy+args.buffer_dimension]
        
    # Compute primitives
    std_conv = std_convoluted(crop_im[args.nir_band - 1].astype(float), 5, args.filter_texture)
    io_utils.save_image(std_conv, "conv5.tif", args.crs, args.transform, 0, args.rpc)

    primitives = np.concatenate(( [crop_ndvi], [crop_ndwi], [std_conv] ), axis=0 )
    io_utils.save_image_n_bands(primitives, "primitives.tif", args.crs, args.transform, 0, args.rpc)
    
    # Segmentation
    if args.segmentation_mode == "NDVI":
        compute_segmentation(args, crop_ndvi, 4e-4, segmentation, primitives)
    else:
        compute_segmentation(args, crop_im, 50, segmentation, primitives)
    
    
    sys.exit(-1) 

    return t_prim, t_seg, segmentation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("im", help="input image (reflectances TOA)")
    parser.add_argument("file_classif", help="Output classification filename")

    parser.add_argument("-red", "--red_band", type=int, nargs="?", default=1, help="Red band index")
    parser.add_argument("-green", "--green_band", type=int, nargs="?", default=2, help="Green band index")
    parser.add_argument("-nir", "--nir_band", type=int, nargs="?", default=4, help="Near Infra-Red band index")
    parser.add_argument("-ndvi", default=None, required=False, action="store", dest="file_ndvi", help="NDVI filename (computed if missing option)")
    parser.add_argument("-ndwi", default=None, required=False, action="store", dest="file_ndwi", help="NDWI filename (computed if missing option)")
    parser.add_argument("-save", choices=["none", "prim", "aux", "all", "debug"], default="none", required=False, action="store", dest="save_mode", help="Save all files (debug), only primitives (prim), only pekel and hand (aux), primitives, pekel and hand (all) or only output mask (none)")
    
    parser.add_argument("-seg", "--segmentation_mode", default="NDVI", help="Image to segment : RGB or NDVI")
    parser.add_argument("-algo_seg", "--algo_seg", choices=["slic", "felz"], default="slic", required=False, action="store", help="Use SkImage SLIC algorithm (slic) or SkImage Felzenszwalb algorithm (felz) for segmentation")
    parser.add_argument("-slic_seg_size", "--slic_seg_size", type=int, default=100, help="Approximative segment size (100 by default)")
    parser.add_argument("-slic_compactness", "--slic_compactness", type=float, default=0.1, help="Balance between color and space proximity (see skimage.slic documentation) - 0.1 by default")
    parser.add_argument("-felz", "--felzenszwalb", default=False, help="Use SkImage Felzenszwalb algorithm for segmentation")
    parser.add_argument("-felz_scale", "--felzenszwalb_scale", type=float, default=1.0, help="Scale parameter for Felzenszwalb algorithm")
    #parser.add_argument("-ref", "--reference_data", nargs="?",
    #                    help="Compute a confusion matrix with this reference data")
    #parser.add_argument("-spth", "--spectral_threshold", type=float, nargs="?", help="Spectral threshold for texture computaton")
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
    #parser.add_argument("-max_workers", "--max_workers", type=int, default=8, help="Max workers for multiprocessed tasks (primitives+segmentation)")
    parser.add_argument("-mask_slic_bool", "--mask_slic_bool", default=False,
                        help="Boolean value wether to use a mask during slic calculation or not")
    parser.add_argument("-mask_slic_file", "--mask_slic_file", help="Raster mask file to use if mask_slic_bool==True")
    parser.add_argument("-no_clean", "--no_clean", default=False, help="Keep temporary files")
    args = parser.parse_args()
    print("DBG > arguments parsed "+str(args))
    
    t0 = time.time()

    primitives = "primitives.tif"
    segmentation = "segmentation.shp"
    
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
    
    segmentation_task(args, startx, starty, im_ndvi, im_ndwi)
    
    
if __name__ == "__main__":
    main()