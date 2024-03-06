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


def accumulate(res_seg, ndvi, ndwi, texture, transform):
    """Stats calculation
    Get the mean of each polygon of the segmentation for NDVI, NDWI and texture bands.
    """
    t0 = time.time()
    # Init
    nb_polys = np.unique(res_seg).size
    counter = np.zeros(nb_polys)
    accumulator_ndvi = np.zeros(nb_polys)
    accumulator_ndwi = np.zeros(nb_polys)
    accumulator_texture = np.zeros(nb_polys)    
    
    # Parse the image and set counter and sum for each polygon
    nb_rows, nb_cols = res_seg.shape
    for r in range(nb_rows):
        for c in range(nb_cols):
            value = res_seg[r][c] - 1
            counter[value] += 1
            accumulator_ndvi[value] += ndvi[r][c]
            accumulator_ndwi[value] += ndwi[r][c]
            accumulator_texture[value] += texture[r][c]
    
    # Get means for each polygon
    for value in range(nb_polys):
        accumulator_ndvi[value] /= counter[value]
        accumulator_ndwi[value] /= counter[value]
        accumulator_texture[value] /= counter[value]

    # Get geometry of each polygon
    geometry = nb_polys*[0]
    for geom, val in rasterio.features.shapes(res_seg.astype("int16"), transform=transform):
        geometry[int(val)-1] = shape(geom)
   
    # Compute dataframe
    datas = {
        "geometry": geometry,
        "count": counter,
        "mean_ndvi": accumulator_ndvi,
        "mean_ndwi": accumulator_ndwi,
        "mean_texture": accumulator_texture
    }
    return pd.DataFrame(datas), (time.time() - t0)
    

def compute_segmentation(args, img, mask, ndvi, ndwi, texture, transform):
    """Compute segmentation with SLIC or Felzenszwalb method"""
    t0 = time.time()
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
    t_seg = time.time() - t0
    
    # Stats calculation    
    df_stats, t_stats = accumulate(res_seg, ndvi, ndwi, texture, transform)

    return df_stats, t_seg, t_stats


def apply_clustering(args, gdf):
    t0 = time.time()
    # Extract NDVI and NDWI2 mean values of each segment
    radiometric_indices = np.stack((gdf.mean_ndvi.values, gdf.mean_ndwi.values), axis=1) 

    # Note : the seed for random generator is fixed to obtain reproductible results
    print("K-Means on radiometric indices : "+str(len(radiometric_indices))+" elements")
    kmeans_rad_indices = KMeans(n_clusters=9,
                                 init="k-means++",
                                 n_init=5,
                                 verbose=0,
                                 random_state=712)
    pred_veg = kmeans_rad_indices.fit_predict(radiometric_indices)
    print(kmeans_rad_indices.cluster_centers_)

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
                

    gdf["pred_veg"] = apply_map(pred_veg, map_centroid)

    figure_name = splitext(args.file_classif)[0] + "_centroids_veg.png"
    display_clusters(list_clusters, "ndvi", "ndwi", nb_clusters_no_veg, (9-nb_clusters_veg), figure_name)

    # data_textures = np.stack((gdf[gdf.pred_veg==VEG_CODE].min_2.values, gdf[gdf.pred_veg==VEG_CODE].max_2.values), axis=1)
    data_textures = np.transpose(np.nan_to_num([gdf[gdf.pred_veg >= UNDEFINED_VEG].mean_texture.values]))

    print("K-Means on texture : "+str(len(data_textures))+" elements")
    nb_clusters_texture = 9
    kmeans_texture = KMeans(n_clusters=nb_clusters_texture, init="k-means++", verbose=0, random_state=712)
    pred_texture = kmeans_texture.fit_predict(data_textures)
    print(kmeans_texture.cluster_centers_)

    list_clusters = pd.DataFrame.from_records(kmeans_texture.cluster_centers_, columns=['mean_texture'])
    list_clusters_by_texture = list_clusters.sort_values(by='mean_texture', ascending=False).index

    map_centroid = []
    nb_clusters_high_veg = int(kmeans_texture.n_clusters / 3)
    for t in range(kmeans_texture.n_clusters):
        if t in list_clusters_by_texture[:nb_clusters_high_veg]:
            map_centroid.append(HIGH_VEG_CODE)
        elif t in list_clusters_by_texture[nb_clusters_high_veg:2*nb_clusters_high_veg]:
            map_centroid.append(UNDEFINED_TEXTURE)
        else:
            map_centroid.append(LOW_VEG_CODE)

    figure_name = splitext(args.file_classif)[0] + "_centroids_texture.png"
    display_clusters(list_clusters, "mean_texture", "mean_texture", nb_clusters_high_veg,
                     2*nb_clusters_high_veg, figure_name)

    gdf["Texture"] = 0
    gdf.loc[gdf.pred_veg >= UNDEFINED_VEG, "Texture"] = apply_map(pred_texture, map_centroid)

    # Ex : 10 (undefined) + 3 (textured) -> 13
    gdf["ClasseN"] = gdf["pred_veg"] + gdf["Texture"]
    
    t1 = time.time()
    extension = splitext(args.file_classif)[1]
    if extension == ".tif":
        print("DBG > Rasterize output -> "+str(args.file_classif))
        # Compressed .tif ouptut
        im = rasterio.open(args.im)
        meta = im.meta.copy()
        meta.update(compress='lzw', driver='GTiff')
        with rasterio.open(args.file_classif, 'w+', **meta) as out:
            out_arr = out.read(1)
            shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf.ClasseN))
            burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
            out.write_band(1, burned)
    else:
        # supposed to be a vector ouput
        fiona_driver ='ESRI Shapefile'        
        if extension == ".gpkg":
            fiona_driver = "GPKG"
        elif extension == ".geojson":
            fiona_driver = "GeoJSON"
        
        print("DBG > driver used "+str(fiona_driver)+ " extension = ["+str(extension)+"]")
        gdf.to_file(args.file_classif, driver=fiona_driver)
    t2 = time.time()

    return (t1-t0), (t2-t1)
    

def segmentation_task(args, im_phr, im_ndvi, im_ndwi, mask_slic, transform, beginx, beginy):                
    # Compute textures
    texture, t_texture = std_convoluted(im_phr[args.nir_band - 1].astype(float), 5, args.filter_texture)
    if (args.save_mode != "none" and args.save_mode != "prim"):
        io_utils.save_image(
            texture,
            join(dirname(args.file_classif), "texture_" + str(beginx) + "_" + str(beginy) + ".tif"),
            args.crs,
            transform,
            nodata=32767,
            rpc=args.rpc
        )
    # Segmentation
    df, t_seg, t_stats = compute_segmentation(args, im_phr, mask_slic, im_ndvi, im_ndwi, texture, transform)
    return df, t_texture, t_seg, t_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("im", help="input image (reflectances TOA)")
    parser.add_argument("file_classif", help="Output classification filename")

    parser.add_argument("-red", "--red_band", type=int, nargs="?", default=1, help="Red band index")
    parser.add_argument("-green", "--green_band", type=int, nargs="?", default=2, help="Green band index")
    parser.add_argument("-nir", "--nir_band", type=int, nargs="?", default=4, help="Near Infra-Red band index")
    parser.add_argument("-ndvi", default=None, required=False, action="store", dest="file_ndvi", help="NDVI filename (computed if missing option)")
    parser.add_argument("-ndwi", default=None, required=False, action="store", dest="file_ndwi", help="NDWI filename (computed if missing option)")
    parser.add_argument("-texture", "--filter_texture", type=int, default=98, help="Percentile for texture (between 1 and 99)")
    parser.add_argument("-save", choices=["none", "prim", "aux", "all", "debug"], default="none", required=False, action="store", dest="save_mode", help="Save all files (debug), only primitives (prim), only shp files (aux), primitives and shp files (all) or only output mask (none)")
    
    parser.add_argument("-seg", "--segmentation_mode", choices=["RGB", "NDVI"], default="NDVI", help="Image to segment : RGB or NDVI")
    parser.add_argument("-algo_seg", "--algo_seg", choices=["slic", "felz"], default="slic", required=False, action="store", help="Use SkImage SLIC algorithm (slic) or SkImage Felzenszwalb algorithm (felz) for segmentation")
    parser.add_argument("-slic_seg_size", "--slic_seg_size", type=int, default=100, help="Approximative segment size (100 by default)")
    parser.add_argument("-slic_compactness", "--slic_compactness", type=float, default=0.1, help="Balance between color and space proximity (see skimage.slic documentation) - 0.1 by default")
    parser.add_argument("-felz", "--felzenszwalb", default=False, help="Use SkImage Felzenszwalb algorithm for segmentation")
    parser.add_argument("-felz_scale", "--felzenszwalb_scale", type=float, default=1.0, help="Scale parameter for Felzenszwalb algorithm")
    
    parser.add_argument("-nbclusters", "--nb_clusters_veg", type=int, default=3, help="Nb of clusters considered as vegetaiton (1-9), default : 3")
    parser.add_argument("-min_ndvi_veg","--min_ndvi_veg", type=int, help="Minimal mean NDVI value to consider a cluster as vegetation (overload nb clusters choice)")
    parser.add_argument("-max_ndvi_noveg","--max_ndvi_noveg", type=int, help="Maximal mean NDVI value to consider a cluster as non-vegetation (overload nb clusters choice)")
    parser.add_argument("-non_veg_clusters","--non_veg_clusters", default=False, required=False, action="store_true", 
                        help="Labelize each 'non vegetation cluster' as 0, 1, 2 (..) instead of single label (0)")
    
    parser.add_argument("-startx", "--startx", type=int, default=0, help="Start x coordinates (crop ROI)")
    parser.add_argument("-starty", "--starty", type=int, default=0, help="Start y coordinates (crop ROI)")
    parser.add_argument("-sizex", "--sizex", type=int, help="Size along x axis (crop ROI)")
    parser.add_argument("-sizey", "--sizey", type=int, help="Size along y axis (crop ROI)")
    parser.add_argument("-buffer", "--buffer_dimension", type=int, default=512, help="Buffer dimension")
    parser.add_argument("-n_workers", "--nb_workers", type=int, default=8, help="Number of workers for multiprocessed tasks (primitives+segmentation)")
    parser.add_argument("-mask_slic_bool", "--mask_slic_bool", default=False,
                        help="Boolean value wether to use a mask during slic calculation or not")
    parser.add_argument("-mask_slic_file", "--mask_slic_file", help="Raster mask file to use if mask_slic_bool==True")
    args = parser.parse_args()
    print("DBG > arguments parsed "+str(args))
    
    t0 = time.time()
    
    image = args.im
    buffer_dimension = args.buffer_dimension
    
    ds_phr = rasterio.open(image)
    
    args.crs = ds_phr.crs
    args.transform = ds_phr.transform
    args.rpc = ds_phr.tags(ns="RPC")
    im_height = ds_phr.height
    im_width = ds_phr.width
    
    im_phr = ds_phr.read()
    
    ds_phr.close()
    
    
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
    stopx = im_height
    starty = 0
    stopy = im_width

    if args.sizex:
        startx = max(args.startx, 0)
        stopx = min(startx + args.sizex, im_height)
    if args.sizey:
        starty = max(args.starty, 0)
        stopy = min(starty + args.sizey, im_width)
    
    if args.mask_slic_bool =="True":
        ds_mask = rasterio.open(args.mask_slic_file)
        print_dataset_infos(ds_mask, "SLIC MASK")
        mask_slic = ds_mask.read(1).astype(bool)
        ds_mask.close()
    else:
        mask_slic = None
        
    ## Parallelisation avec concurrent (to remplace with eoscale)
    time_texture = 0
    time_segmentation = 0
    time_stats = 0
    cptx = 0
    cpty = 0
    future_seg = []
    list_res = []
    
    t0_texture_seg_stats = time.time()

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
                    get_transform(args.transform, beginx, beginy),
                    beginx,
                    beginy
                ))
                cpty += 1
            cptx += 1
            cpty = 0
        
        
        for seg in concurrent.futures.as_completed(future_seg):
            try:
                df_res, t_texture, t_seg, t_stats = seg.result()
                list_res.append(df_res)
                    
            except Exception as e:
                print("Exception ---> "+str(e))
            else:
                time_texture += t_texture
                time_segmentation += t_seg
                time_stats += t_stats

    df_total = pd.concat(list_res, ignore_index=True)    
    gdf_total = gpd.GeoDataFrame(df_total, geometry='geometry', crs=args.crs)
    
    if args.save_mode == "debug":
        cols = df_total.columns.tolist()
        df_total["polygon_id"] = np.arange(1, len(df_total.index) + 1)
        df_total = df_total[["polygon_id"] + cols]
        gdf_total = gpd.GeoDataFrame(df_total, geometry='geometry', crs=args.crs)
        gdf_total.to_file("segmentation.shp")
        
    delay_texture_seg_stats = time.time() - t0_texture_seg_stats
    
    # Clustering
    time_clustering, time_io = apply_clustering(args, gdf_total)
    
    # A aligner sur les autres masques
    print("**** Vegetation mask for "+str(args.im)+" (saved as "+str(args.file_classif)+") ****")
    print("Total time (user)       :\t"+str(time.time()-t0))
    print("Delay for texture + segmentation : "+str(delay_texture_seg_stats))
    print("Max workers used for parallel tasks "+str(args.nb_workers))
    print("Texture (parallel)   :\t"+str(time_texture))
    print("Segmentation (parallel) :\t"+str(time_segmentation))
    print("Statistics (parallel) :\t"+str(time_stats))
    print("Clustering              :\t"+str(time_clustering))    
    print("Writing output file     :\t"+str(time_io))
    
    
if __name__ == "__main__":
    main()