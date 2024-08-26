#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Use a global land cover map to calculate the better number of vegetation cluster to use for mask computation"""
# ligne test :
# python analyse_glcm.py /work/CAMPUS/etudes/Masques_CO3D/Data/ClassifRef/WORLDCOVER/esa_worldcover.vrt /work/CAMPUS/etudes/Masques_CO3D/Data/Images/Toulouse/xt_PHR_uint16.tif

import traceback
import argparse
import json

import numpy as np

import otbApplication as otb


def get_advices(veg, low_veg, high_veg, nb_total):
    
    """ Return adviced number of clusters for vegetation and high vegetation regarding ratio of these classes in the image """
    pct_veg = 100*veg/nb_total
    if pct_veg == 0:
        return 0, 0
    nb_clusters_veg = 3
    if pct_veg < 5:
        nb_clusters_veg = 1
    elif pct_veg < 25:
        nb_clusters_veg = 2
    elif 60 < pct_veg <= 85:
        nb_clusters_veg = 4
    elif pct_veg > 85:
        nb_clusters_veg = 5

    nb_clusters_low_veg = 3
    pct_low_veg = 100 * low_veg / (low_veg + high_veg)

    if pct_low_veg < 5:
        nb_clusters_low_veg = 1
    elif pct_low_veg < 25:
        nb_clusters_low_veg = 2
    elif 60 < pct_low_veg <= 85:
        nb_clusters_low_veg = 4
    elif pct_low_veg > 85:
        nb_clusters_low_veg = 5

    return nb_clusters_veg, nb_clusters_low_veg
    
        
def compute_stats(args):
    """ Compute ratio of vegetation, low vegetation and vegetation in the ROI  """
    
    app_roi = otb.Registry.CreateApplication("ExtractROI")
    params_roi = {"out":"fake.tif","mode":"fit","mode.fit.im":args.im,"in":args.map}
    app_roi.SetParameters(params_roi)
    app_roi.Execute()
    data_map = app_roi.GetVectorImageAsNumpyArray("out")
    print(f"{data_map.shape}")

    legend = {10:"Tree cover" ,
            20:"Shrubland" ,
            30:"Grassland" ,
            40:"Cropland",
            50:"Built-up",
            60:"Bare / Sparse vegetation" ,
            70:"Snow and ice" ,
            80:"Permanent water bodies" ,
            90:"Herbaceous wetland",
            95:"Mangroves",
            100:"Moss and lichen"
             }
    width = data_map.shape[0]
    height = data_map.shape[1]

    nb_total = width * height
    unique, counts = np.unique(data_map, return_counts=True)

    veg, low_veg, high_veg = 0,0,0
    for v, c in zip(unique, counts):
        print(f"{v} : {c} pixels ({100*c/nb_total:.1f}%) - class {legend[v]}")
        if v in [10, 20, 30, 40, 90, 95, 100]:
            veg += c

        if v in [20, 30, 40, 90, 100 ]:
            low_veg += c

        if v in [10, 95]:
            high_veg += c

    nb_clusters_veg, nb_clusters_low_veg = get_advices(veg, low_veg, high_veg, nb_total)
            
    print(f"Vegetation (% area) \t: {100*veg/nb_total:.2f}%")
    print(f"Low vegetation (% area) \t: {100*low_veg/nb_total:.2f}%")
    print(f"High vegetation (% area) \t: {100*high_veg/nb_total:.2f}%")

    print(f"export VEG_CLUSTERS={nb_clusters_veg}")
    print(f"export LOW_VEG_CLUSTERS={nb_clusters_low_veg}")
    
    if args.save is not None:
        app_s_impose = otb.Registry.CreateApplication("Superimpose")
        params_s_impose = {"inr":args.im, "inm":args.map, "out":str(args.save), "interpolator":"nn"}
        app_s_impose.SetParameters(params_s_impose)
        app_s_impose.ExecuteAndWriteOutput()
    
    return nb_clusters_veg, nb_clusters_low_veg
        
def update_json(config_file, nb_clusters_veg, nb_clusters_low_veg):
    """ If 'config_file_to_update' is passed, overwrite  nb_clusters_veg and nb_clusters_low_veg arguments with the value adviced """
    
    with open(config_file, "r", encoding="utf8") as file:
        dict_config=json.load(file)
        dict_config["vegetation"].update({"nb_clusters_veg": nb_clusters_veg,
                                          "nb_clusters_low_veg": nb_clusters_low_veg })
    with open(config_file, "w", encoding="utf8") as file:
        json.dump(dict_config, file)
    
    
def getarguments():
    """ Parse command line arguments. """

    parser = argparse.ArgumentParser(description="Compute stats on a global land cover map")

    parser.add_argument("map", help="Input land cover map")
    parser.add_argument("im", help="Input image : will crop and compute stats over this region of interest")
    parser.add_argument("-config_file_to_update", type=str, default="None", help= "Update the VEG_CLUSTERS and LOW_VEG_CLUSTERS parameters in the JSON file (create them if they don't exist)")
    parser.add_argument(
         "-save",
         default=None,
         required=False,
         action="store",
         dest="save",
         help="Crop and save input land cover map"
     )
     
    return parser.parse_args()


def analyse_glcm(map, im, config_file_to_update):
    """ Main function """ 
    try:
        nb_clusters_veg, nb_clusters_low_veg = compute_stats(map, im)
        
        if config_file_to_update != "None":
            update_json(config_file_to_update, nb_clusters_veg, nb_clusters_low_veg)

    except FileNotFoundError as fnfe_exception:
        print("FileNotFoundError", fnfe_exception)

    except PermissionError as pe_exception:
        print("PermissionError", pe_exception)

    except ArithmeticError as ae_exception:
        print("ArithmeticError", ae_exception)

    except MemoryError as me_exception:
        print("MemoryError", me_exception)

    except Exception as exception: 
        print("oups...", exception)
        traceback.print_exc()
        