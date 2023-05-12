#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import traceback
import argparse
import time
from os.path import dirname, join

from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, f1_score, precision_score, recall_score
import rasterio as rio
from rasterio import features
import geopandas as gpd
from shapely.geometry import shape
import numpy as np
from slum.tools import io_utils

import sys


def generate_polygons_in_gdf(im, crs, transform):
    buildings = np.array(list(features.shapes(im, transform=transform)))
    mask = (buildings[:, 1] == 1.0)
    buildings = buildings[mask]
    #print(buildings.shape)
    geometry = np.array([shape(buildings[i][0]) for i in range(0, len(buildings))])
    gdf = gpd.GeoDataFrame(
        {'id': list(range(1, len(geometry)+1)), 'geometry': geometry },
        crs=crs
    )
    return gdf


def buildings_count(args, im_ref, im_predict, dir_out, crs_ref, transform_ref, crs_predict, transform_predict):
    start_time = time.time()
    
    # Generate GeoDataFrame
    gdf_predict = generate_polygons_in_gdf(im_predict, crs_predict, transform_predict)
    gdf_ref = generate_polygons_in_gdf(im_ref, crs_ref, transform_ref)
    
    if args.unit != "meter":
        # change spacing unit in meter
        crs_ref = gdf_ref.estimate_utm_crs()
        gdf_ref = gdf_ref.to_crs(crs_ref)
    
    # Remove small builidings from ground truth
    gdf_ref_filtered = gdf_ref[gdf_ref.geometry.area > args.area]
    
    if args.save:  
        gdf_predict.to_file(join(dir_out, "buildings_predict.shp"))
        gdf_ref_filtered.to_file(join(dir_out, "buildings_ref.shp"))

    # Intersection calculation
    if crs_ref != crs_predict:
        gdf_ref_filtered = gdf_ref_filtered.to_crs(crs_predict)

    gdf_intersect = gpd.overlay(gdf_ref_filtered, gdf_predict, how="intersection", keep_geom_type=True)
    
    print("Detected buildings : " + str(gdf_intersect["id_1"].nunique()) + "/" + str(gdf_ref_filtered.shape[0]))
    print("Buildings count execution time : "+str(time.time() - start_time)) 


def get_merged_image(im_ref, im_predict, path_out, crs, transform, rpc):
    start_time = time.time()
    im_merged = np.add(im_ref, 2*im_predict)
    
    io_utils.save_image(
            im_merged,
            path_out,
            crs,
            transform,
            255,
            rpc,
        )

    print("Merge execution time : "+str(time.time() - start_time))  


def get_score(im_ref, im_predict):
    start_time = time.time()
    print("Accuracy >>>", accuracy_score(im_ref, im_predict)) 
    precision = precision_score(im_ref, im_predict)
    print("Precision >>>", precision) 
    recall = recall_score(im_ref, im_predict)
    print("Recall >>>", recall)
    f1 = 2 * precision * recall / (precision + recall)
    print("F1 >>>", f1)
    
    # print("Log loss >>>", log_loss(im_ref, im_predict))
    # print("Confusion matrix >>>", confusion_matrix(im_ref, im_predict))
    #print("F1 >>>", f1_score(im_ref, im_predict))
    
    print("Scores calculation execution time : "+str(time.time() - start_time))


def getarguments():
    """ Parse command line arguments. """

    parser = argparse.ArgumentParser(description='Rasterize OSM layer with respect to an input image geographic extent and spacing')

    parser.add_argument('-gt', required=True, action='store', dest='gt',
                        help='Ground truth file (OSM, WSF...)')
    parser.add_argument('-im', required=True, action='store', dest='im',
                       help='Prediction image')
    parser.add_argument('-out', required=True, action='store', dest='out',
                       help='Output filename')
    parser.add_argument('-value_classif', required=False, action='store', dest='value_classif', type=int, default=1,
                       help='Ground truth classification value (default is 1)')
    parser.add_argument('-polygonize', required=False, action='store_true', dest='polygonize', default=False,
                       help='Will estimate the number of buildings from the truth file predicted')
    parser.add_argument('-polygonize.area', required=False, action='store', dest='area', type=int, default=0,
                       help='Minimal area required in the ground truth file (default is 0)')
    parser.add_argument('-polygonize.unit', required=False, action='store', dest='unit', choices=["meter", "degree"], default="meter",
                       help='Unit of spacing (default is meter)')
    parser.add_argument('-save', required=False, action='store_true', dest='save', default=False,
                       help='Save SHP files containing the buildings')

    return parser.parse_args()


def main():
    try:
        args = getarguments()
        
        # Get ground truth
        ds_ref = rio.open(args.gt)
        crs_ref = ds_ref.crs
        transform_ref = ds_ref.transform
        im_ref = ds_ref.read(1)
        ds_ref.close()
        del ds_ref

        if args.value_classif != 1:
            im_ref[im_ref != args.value_classif] = 0
            im_ref[im_ref == args.value_classif] = 1

        # Get predicted mask
        ds_predict = rio.open(args.im)
        crs_predict = ds_predict.crs
        transform_predict = ds_predict.transform
        rpc = ds_predict.tags(ns="RPC")
        im_predict = ds_predict.read(1)
        ds_predict.close()
        del ds_predict 

        im_predict[im_predict > 1] = 0  # for urbanmask_seg
        
        # Count buildings
        if args.polygonize:
            buildings_count(
                args, 
                im_ref, 
                im_predict, 
                dirname(args.out),
                crs_ref, 
                transform_ref,
                crs_predict,
                transform_predict
            )
        
        # Create merge image
        get_merged_image(im_ref, im_predict, args.out, crs_predict, transform_predict, rpc)
        
        # Transform in 1d array
        im_predict_1d = im_predict.flatten()
        im_ref_1d = im_ref.flatten()
        
        # Scores calculation
        get_score(im_ref_1d, im_predict_1d)

    except FileNotFoundError as fnfe_exception:
        print('FileNotFoundError', fnfe_exception)

    except PermissionError as pe_exception:
        print('PermissionError', pe_exception)

    except ArithmeticError as ae_exception:
        print('ArithmeticError', ae_exception)

    except MemoryError as me_exception:
        print('MemoryError', me_exception)

    except Exception as exception: # pylint: disable=broad-except
        print('oups...', exception)
        traceback.print_exc()

if __name__ == '__main__':
    main()
