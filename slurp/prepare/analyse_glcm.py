#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of SLURP
# (see https://github.com/CNES/slurp).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" 
Use a global land cover map to calculate the better number of vegetation cluster to use for mask computation
"""

import traceback

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
    
        
def compute_stats(map_lc, im):
    """ Compute ratio of vegetation, low vegetation and vegetation in the ROI  """
    
    app_roi = otb.Registry.CreateApplication("ExtractROI")
    params_roi = {"out":"fake.tif","mode":"fit","mode.fit.im":im,"in": map_lc}
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

    
    return nb_clusters_veg, nb_clusters_low_veg


def analyse_glcm(map_lc, im):
    """ Main function """ 
    try:
        nb_clusters_veg, nb_clusters_low_veg = compute_stats(map_lc, im)
        
        return nb_clusters_veg, nb_clusters_low_veg

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
        