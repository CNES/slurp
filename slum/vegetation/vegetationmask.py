#!/usr/bin/python
import sys
import glob
import os
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
import matplotlib.pyplot as plt
import time
import otbApplication as otb
from skimage.segmentation import slic
from skimage.segmentation import felzenszwalb
import concurrent.futures

NO_VEG_CODE = 0
WATER_CODE = 3

LOW_VEG_CODE = 1
UNDEFINED_TEXTURE = 2
HIGH_VEG_CODE = 3

UNDEFINED_VEG = 10
VEG_CODE = 20

# Max width/height we open at once to compute spectral threlshold
MAX_BUFFER_SIZE = 5000 


def compute_primitives(args, image, primitives, spectral_threshold):
    """
    Computes
    """
    t0 = time.time()
    app_rindices = otb.Registry.CreateApplication("RadiometricIndices")
    app_rindices.SetParameterString("in", image)
    app_rindices.SetParameterString("out", "radiometric_indices.tif")

    blue = 3
    green = 2
    if args.red_band == 3:
        blue = 1

    app_rindices.SetParameterValue("channels.blue", blue)
    app_rindices.SetParameterValue("channels.green", green)
    app_rindices.SetParameterValue("channels.red", args.red_band)
    app_rindices.SetParameterValue("channels.nir", args.nir_band)

    app_rindices.SetParameterStringList("list", ["Vegetation:NDVI", "Water:NDWI2"])
    app_rindices.Execute()

    app_sfs_texture = otb.Registry.CreateApplication("SFSTextureExtraction")
    app_sfs_texture.SetParameterString("in", image)
    app_sfs_texture.SetParameterInt("channel", 4)
    app_sfs_texture.SetParameterFloat("parameters.spethre", float(spectral_threshold))
    app_sfs_texture.SetParameterInt("parameters.spathre", 15)
    app_sfs_texture.SetParameterInt("parameters.nbdir", 10)
    app_sfs_texture.SetParameterValue("out", "sfs.tif")
    app_sfs_texture.Execute()

    app_concat = otb.Registry.CreateApplication("ConcatenateImages")
    app_concat.AddImageToParameterInputImageList("il", app_rindices.GetParameterOutputImage("out"))
    app_concat.AddImageToParameterInputImageList("il", app_sfs_texture.GetParameterOutputImage("out"))
    app_concat.SetParameterString("out", primitives)

    app_concat.ExecuteAndWriteOutput()
    return (time.time() - t0)


def save_image(image, file, crs=None, transform=None, nodata=None, rpc=None, **kwargs):
    """ Save 1 band numpy image to file with lzw compression.
    Note that rio.dtype is string so convert np.dtype to string.
    rpc must be a dictionnary.
    """
    with rasterio.open(file,
                  'w',
                  driver='GTiff',
                  compress='deflate',
                  height=image.shape[0],
                  width=image.shape[1],
                  count=1,
                  dtype="int16",
                  crs=crs,
                  transform=transform,
                  **kwargs) as dataset:
        dataset.write(image, 1)
        dataset.nodata = nodata

        if rpc:
            dataset.update_tags(**rpc, ns='RPC')

        dataset.close()


def compute_segmentation(args, image, range, segmentation, primitives):
    t0 = time.time()
    segmentation_raster = segmentation.replace(".shp", ".tif")
    app_LSMS = ""

    if args.slic:
        print("DBG > compute_segmentation (skimage SLIC)" + str(primitives))
        if args.segmentation_mode == "RGB":
             with rasterio.open(image) as ds_img :
                # Note : we read RGB image.
                img = ds_img.read()
                nseg = int(img.shape[2] * img.shape[1] / args.slic_seg_size)
                data = img.reshape(img.shape[1], img.shape[2], img.shape[0])[:,:,:3]

                res_slic = slic(data, compactness=float(args.slic_compactness), n_segments=nseg, sigma=1, convert2lab=True, channel_axis = 2)
                save_image(res_slic.astype("int16"), segmentation_raster, crs=ds_img.crs, transform=ds_img.transform,
                     rpc=ds_img.tags(ns='RPC'), nodata=0)
        else:
            with rasterio.open(primitives) as ds_img :
                # Note : we read primitives. NDVI is the first band
                # TODO : implement segmentation on RGB images, with img.reshape...
                img = ds_img.read()
                ndvi = 1000 * img[1, :, :]
                print("ndvi.shape = ", ndvi.shape)
                # Estimation of the max number of segments (ie : each segment is > 100 pixels)
                nseg = int(img.shape[2] * img.shape[1] / args.slic_seg_size)

                if args.mask_slic_bool == "True" :
                    # Open mask file with good shape
                    with rasterio.open(args.mask_slic_file) as ds_mask:
                        mask=ds_mask.read(1)
                        mask = mask.astype(bool)
                        res_slic = slic(ndvi.astype("double"), compactness=float(args.slic_compactness), n_segments=nseg, mask=mask, sigma=1)
                else:
                    res_slic = slic(ndvi.astype("double"), compactness=float(args.slic_compactness), n_segments=nseg, sigma=1)

                save_image(res_slic.astype("int16"), segmentation_raster, crs=ds_img.crs, transform=ds_img.transform,
                        rpc=ds_img.tags(ns='RPC'), nodata=0)
    elif args.felzenszwalb:
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
    else:
        app_LSMS = otb.Registry.CreateApplication("LargeScaleMeanShift")
        app_LSMS.SetParameterString("in", image)
        app_LSMS.SetParameterString("mode", "raster")
        app_LSMS.SetParameterInt("minsize", 200)
        app_LSMS.SetParameterFloat("ranger", range)
        app_LSMS.SetParameterString("mode.raster.out", segmentation_raster)
        app_LSMS.Execute()

    app_ZonalStats = otb.Registry.CreateApplication("ZonalStatistics")
    app_ZonalStats.SetParameterString("in", primitives)
    app_ZonalStats.SetParameterString("inzone", "labelimage")
    if args.slic or args.felzenszwalb:
        app_ZonalStats.SetParameterString("inzone.labelimage.in",segmentation_raster)
    else:
        app_ZonalStats.SetParameterInputImage("inzone.labelimage.in", app_LSMS.GetParameterOutputImage("mode.raster.out"))
    app_ZonalStats.SetParameterString("out.vector.filename", segmentation)
    app_ZonalStats.ExecuteAndWriteOutput()
    return (time.time() - t0)


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
    

def apply_clustering(args, gdf):
    t0 = time.time()
    # Extract NDVI and NDWI2 mean values of each segment
    radiometric_indices = np.stack((gdf.mean_0.values, gdf.mean_1.values), axis=1)

    # Note : the seed for random generator is fixed to obtain reproductible results
    print("K-Means on radiometric indices : "+str(len(radiometric_indices))+" elements")
    kmeans_rad_indices = KMeans(n_clusters=9,
                                 init="k-means++",
                                 n_init=5,
                                 verbose=0,
                                 random_state=712)
    pred_veg = kmeans_rad_indices.fit_predict(radiometric_indices)
    print(kmeans_rad_indices.cluster_centers_)

    list_clusters = pd.DataFrame.from_records(kmeans_rad_indices.cluster_centers_, columns=['ndvi', 'ndwi2'])
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
                # 0
                map_centroid.append(NO_VEG_CODE)
            elif t in list_clusters_by_ndvi[nb_clusters_no_veg:9-args.nb_clusters_veg]:
                # 10
                map_centroid.append(UNDEFINED_VEG)
            else:
                # 20
                map_centroid.append(VEG_CODE)
                

    gdf["pred_veg"] = apply_map(pred_veg, map_centroid)

    figure_name = os.path.splitext(args.out)[0] + "_centroids_veg.png"
    display_clusters(list_clusters, "ndvi", "ndwi2", nb_clusters_no_veg, (9-nb_clusters_veg), figure_name)

    # data_textures = np.stack((gdf[gdf.pred_veg==VEG_CODE].min_2.values, gdf[gdf.pred_veg==VEG_CODE].max_2.values), axis=1)
    data_textures = np.nan_to_num(np.stack(
        (gdf[gdf.pred_veg >= UNDEFINED_VEG].mean_2.values,
         gdf[gdf.pred_veg >= UNDEFINED_VEG].stdev_2.values), axis=1))


    print("K-Means on SFS textures : "+str(len(data_textures))+" elements")
    nb_clusters_texture = 9
    kmeans_texture = KMeans(n_clusters=nb_clusters_texture, init="k-means++", verbose=0, random_state=712)
    pred_texture = kmeans_texture.fit_predict(data_textures)
    print(kmeans_texture.cluster_centers_)

    list_clusters = pd.DataFrame.from_records(kmeans_texture.cluster_centers_, columns=['mean_sfs', 'stdev_sfs'])
    list_clusters_by_texture = list_clusters.sort_values(by='mean_sfs', ascending=True).index

    map_centroid = []
    nb_clusters_high_veg = int(kmeans_texture.n_clusters / 3)
    for t in range(kmeans_texture.n_clusters):
        if t in list_clusters_by_texture[:nb_clusters_high_veg]:
            map_centroid.append(HIGH_VEG_CODE)
        elif t in list_clusters_by_texture[nb_clusters_high_veg:2*nb_clusters_high_veg]:
            map_centroid.append(UNDEFINED_TEXTURE)
        else:
            map_centroid.append(LOW_VEG_CODE)

    figure_name = os.path.splitext(args.out)[0] + "_centroids_texture.png"
    display_clusters(list_clusters, "mean_sfs", "stdev_sfs", nb_clusters_high_veg,
                     2*nb_clusters_high_veg, figure_name)

    gdf["Texture"] = 0
    gdf.loc[gdf.pred_veg >= UNDEFINED_VEG, "Texture"] = apply_map(pred_texture, map_centroid)

    # Ex : 10 (undefined) + 3 (textured) -> 13
    gdf["ClasseN"] = gdf["pred_veg"] + gdf["Texture"]
    
    t1 = time.time()
    extension = os.path.splitext(args.out)[1]
    if extension == ".tif":
        print("DBG > Rasterize output -> "+str(args.out))
        # Compressed .tif ouptut
        im = rasterio.open(args.im)
        meta = im.meta.copy()
        meta.update(compress='lzw')
        with rasterio.open(args.out, 'w+', **meta) as out:
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
        gdf.to_file(args.out, driver=fiona_driver)
    t2 = time.time()

    return (t1-t0), (t2-t1)


def rasterize_and_evaluate(segmentation, image_ref, ground_truth):
    app_raster = otb.Registry.CreateApplication("Rasterization")
    app_raster.SetParameterString("in", segmentation)
    app_raster.SetParameterString("mode", "attribute")
    app_raster.SetParameterString("mode.attribute.field", "ClasseN")
    app_raster.SetParameterString("im", image_ref)
    app_raster.SetParameterInt("background", 0)
    app_raster.SetParameterString("out", "raster.tif")
    app_raster.Execute()

    app_conf_mat = otb.Registry.CreateApplication("ComputeConfusionMatrix")
    app_conf_mat.SetParameterInputImage("in", app_raster.GetParameterOutputImage("out"))
    app_conf_mat.SetParameterString("out", "confmat.txt")
    app_conf_mat.SetParameterString("ref", "vector")
    app_conf_mat.SetParameterString("ref.vector.in", ground_truth)
    app_conf_mat.UpdateParameters()
    # TODO : fix field name : should be case sensitive
    app_conf_mat.SetParameterString("ref.vector.field", "classen")
    app_conf_mat.ExecuteAndWriteOutput()


def crop_image(image, xt_image, startx, starty, sizex, sizey):
    app_roi = otb.Registry.CreateApplication("ExtractROI")
    app_roi.SetParameterString("in", image)
    app_roi.SetParameterString("out", xt_image)
    app_roi.SetParameterInt("startx", startx)
    app_roi.SetParameterInt("starty", starty)
    app_roi.SetParameterInt("sizex", sizex)
    app_roi.SetParameterInt("sizey", sizey)
    app_roi.ExecuteAndWriteOutput()

def compute_primitives_and_segmentation(args, image, primitives, spectral_threshold, segmentation, mask):
    t_prim = compute_primitives(args, image, primitives, spectral_threshold)
    t_seg = 0
    if args.segmentation_mode == "NDVI":
        t_seg = compute_segmentation(args, primitives + "?&bands=1", 4e-4, segmentation, primitives)
    else:
        t_seg = compute_segmentation(args, image, 50, segmentation, primitives)
    return t_prim, t_seg


def instanciate_result(result, clustering):
    try:
        subprocess.call(["ogr2ogr", result, clustering])
        subprocess.call(["rm", clustering.replace(".shp", ".*")])
        print("Creating ", result)
    except:
        print("Error in creating shapefile")
        exit(-1)


def update_result(result, clustering):
    try:
        print("ogrmerge.py -o ", result, clustering, "-single -append -field_strategy Union")
        subprocess.call(["ogrmerge.py -o ", result, clustering, "-single -append -field_strategy Union"])
        subprocess.call(["rm", clustering.replace(".shp", ".*")])
        print("Creating ", result)
    except:
        print("Error in updating shapefile")
        # exit(-1)

def set_threshold_from_random_samples(image, nodata, nb_sample):
    width = image.shape[0]
    height = image.shape[1]
    mask = ma.masked_equal(image, nodata)
    
    cpt = 0
    vec = []
    while cpt < nb_sample:
        row = np.random.randint(0, height)
        col = np.random.randint(0, width)

        if mask[row, col]:
            vec.append(image[row, col])
            cpt += 1

    threshold = np.std(vec)/3.
    return threshold

def segmentation_task(args, beginx, beginy, spectral_threshold):
    print("Segmentation de "+str(args.im)+ " entre "+str(beginx)+" et "+str(beginx+args.buffer_dimension))
    
    image        = "image_"+str(beginx)+"_"+str(beginy)+".tif"
    primitives   = "primitives_"+str(beginx)+"_"+str(beginy)+".tif"
    segmentation = "segmentation_"+str(beginx)+"_"+str(beginy)+".shp"

    crop_image(args.im, image, beginx, beginy, args.buffer_dimension, args.buffer_dimension)

    mask = "mask_" + str(beginx) + "_" + str(beginy) + ".tif"
    
    if args.mask_slic_bool =="True":
        print("Crop of the mask between " + str(beginx) + " and " + 
              str(beginx + args.buffer_dimension) + " / " + str(beginy) + 
              " and " + str(beginy + args.buffer_dimension))
        crop_image(args.mask_slic_file, mask, beginx, beginy, args.buffer_dimension, args.buffer_dimension)
        
    t_prim, t_seg = compute_primitives_and_segmentation(args, image, primitives, spectral_threshold, segmentation, mask)

    return t_prim, t_seg, segmentation


def main():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("im", help="input image (reflectances TOA)")
    parser.add_argument("out", help="segmented mask")

    parser.add_argument("-red", "--red_band", type=int, nargs="?", default=1, help="Red band index")
    parser.add_argument("-nir", "--nir_band", type=int, nargs="?", default=4, help="Near Infra-Red band index")
    parser.add_argument("-seg", "--segmentation_mode", default="NDVI", help="Image to segment : RGB or NDVI")
    parser.add_argument("-slic", "--slic", default=False, help="Use SkImage SLIC algorithm for segmentation")
    parser.add_argument("-slic_seg_size", "--slic_seg_size", type=int, default=100, help="Approximative segment size (100 by default)")
    parser.add_argument("-slic_compactness", "--slic_compactness", type=float, default=0.1, help="Balance between color and space proximity (see skimage.slic documentation) - 0.1 by default")
    parser.add_argument("-felz", "--felzenszwalb", default=False, help="Use SkImage Felzenszwalb algorithm for segmentation")
    parser.add_argument("-felz_scale", "--felzenszwalb_scale", type=float, default=1.0, help="Scale parameter for Felzenszwalb algorithm")
    parser.add_argument("-ref", "--reference_data", nargs="?",
                        help="Compute a confusion matrix with this reference data")
    parser.add_argument("-spth", "--spectral_threshold", type=float, nargs="?", help="Spectral threshold for texture computaton")
    parser.add_argument("-nbclusters", "--nb_clusters_veg", type=int, default=3, help="Nb of clusters considered as vegetaiton (1-9), default : 3")
    parser.add_argument("-min_ndvi_veg","--min_ndvi_veg", type=float, help="Minimal mean NDVI value to consider a cluster as vegetation (overload nb clusters choice)")
    parser.add_argument("-max_ndvi_noveg","--max_ndvi_noveg", type=float, help="Maximal mean NDVI value to consider a cluster as non-vegetation (overload nb clusters choice)")
    #parser.add_argument("-input_veg_centroids", "--input_veg_centroids", help="Input vegetation centroids file")
    parser.add_argument("-startx", "--startx", type=int, default=0, help="Start x coordinates (crop ROI)")
    parser.add_argument("-starty", "--starty", type=int, default=0, help="Start y coordinates (crop ROI)")
    parser.add_argument("-sizex", "--sizex", type=int, help="Size along x axis (crop ROI)")
    parser.add_argument("-sizey", "--sizey", type=int, help="Size along y axis (crop ROI)")
    parser.add_argument("-buffer", "--buffer_dimension", type=int, default=512, help="Buffer dimension")
    parser.add_argument("-max_workers", "--max_workers", type=int, default=8, help="Max workers for multiprocessed tasks (primitives+segmentation)")
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
    result = args.out
    spectral_threshold = 0. # shall be assigned later with the 1st pattern
    buffer_dimension = args.buffer_dimension

    ds = rasterio.open(image)

    startx = 0
    stopx = ds.width
    starty = 0
    stopy = ds.height

    if args.sizex:
        startx = max(args.startx, 0)
        stopx = min(startx + args.sizex, ds.width)
    if args.sizey:
        starty = max(args.starty, 0)
        stopy = min(starty + args.sizey, ds.height)

    if args.spectral_threshold:
        spectral_threshold = args.spectral_threshold
        print("Spectral Threshold : "+str(spectral_threshold)+ " user defined")
    else:
        image_nir = ds.read(4)
        im_nir = ""
        if (stopx-startx > MAX_BUFFER_SIZE and stopy-starty > MAX_BUFFER_SIZE ):
            im_nir = image_nir[startx:startx+MAX_BUFFER_SIZE ,starty:starty+MAX_BUFFER_SIZE]
        else:
            im_nir = image_nir[startx:stopx,starty:stopy]
        nb_samples = 2000
        spectral_threshold = set_threshold_from_random_samples(im_nir, ds.nodata, nb_samples)
        print("Spectral Threshold : "+str(spectral_threshold)+ " from "+str(nb_samples)+" samples")
    ds.close()

    # time for computing the spectral threshold
    time_spth = time.time() - t0

    slic_mode = args.slic

    cptx = 0
    cpty = 0
    
    gdf_total = gpd.GeoDataFrame()
    time_primitives=0
    time_segmentation=0
    time_clustering=0


    msk_slic_bool = args.mask_slic_bool
    msk_slic_file = args.mask_slic_file
    print("msk_slic_bool = ", msk_slic_bool)
    print("msk_slic_file = ", msk_slic_file)
    
    time_segmentation = 0
    time_primitives = 0
    future_seg = []
    list_res = []

    t0_prim_seg = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        while (startx + cptx * buffer_dimension < stopx):
            beginx = startx + cptx * buffer_dimension
            while (starty + cpty * buffer_dimension < stopy):
                beginy = starty + cpty * buffer_dimension
                future_seg.append(executor.submit(segmentation_task, args, beginx, beginy, spectral_threshold))
                cpty += 1
            cptx += 1
            cpty = 0
        
        
        for seg in concurrent.futures.as_completed(future_seg):
            t_seg = 0
            try:
                t_prim, t_seg, seg_shapefile = seg.result()
                list_res.append(seg_shapefile)
                    
            except Exception as e:
                print("Exception ---> "+str(e))
            else:
                time_segmentation += t_seg
                time_primitives += t_prim

        print("Primitives and segmentation parallelised on "+str(executor._max_workers)+" workers max")

    cpt = 0 
    for r in list_res:
        if cpt == 0:
            gdf_total = gpd.read_file(r)
        else:
            gdf = gpd.read_file(r)
            gdf_total = pd.concat([gdf_total,gdf], axis=0)
        cpt = cpt + 1

    delay_prim_seg = time.time() - t0_prim_seg

    time_clustering, time_io = apply_clustering(args, gdf_total)
    
    print("**** Vegetation mask for "+str(args.im)+" (saved as "+str(args.out)+") ****")
    print("Total time (user)       :\t"+str(time.time()-t0))
    print("Spectral threshold      :\t"+str(time_spth))
    print("Delay for primitives + segmentation : "+str(delay_prim_seg))
    print("Max workers used for parallel tasks "+str(args.max_workers))
    print("Primitives (parallel)   :\t"+str(time_primitives))
    print("Segmentation (parallel) :\t"+str(time_segmentation))
    print("Clustering              :\t"+str(time_clustering))    
    print("Writing output file     :\t"+str(time_io))    

    if args.no_clean != "True":
        os.system("rm image_*_*.tif segmentation_*_*.* primitives_*_*.tif")
        if args.mask_slic_bool == True:
            os.system("rm mask_*.*")

if __name__ == "__main__":
    main()
