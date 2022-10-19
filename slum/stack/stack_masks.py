#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script stacks existing masks
"""

import argparse
import rasterio as rio
import numpy as np
import argparse
import os

LOW_VEG  = 1
HIGH_VEG = 2
WATER = 3
BUILDINGS = 4
UNDEF_WATER_URBAN = 5 # Prediction of water, but also urban
BARE_GROUND = 6
UNDEF_URBAN_BARE_GROUND = 7 # Prediction of urban but smooth area (could be bare ground)
WATER_PRED = 8 # Prediction of water, but not in Peckel database
SHADOW = 9
ROADS = 10

LOW = 1
HIGH = 2
NOT_CONFIDENT = 0


def stack(args):
    """
    Stacks all the masks given in the parameters
    :param mask: Mask path
    :param vegetation: Path of vegetation mask
    :param water: Path of water mask (classif.tif)
    :param water_pred: Path of water mask (predict.tif)
    :param building: Path of urban mask
    """
    mask_fic_veg = rio.open(args.vegetation)
    mask_veg = mask_fic_veg.read(1)
    profile = mask_fic_veg.profile

    mask_fic_water = rio.open(args.water)
    mask_water = mask_fic_water.read(1)

    mask_fic_water_pred = rio.open(args.water_pred)
    mask_water_pred = mask_fic_water_pred.read(1)

    mask_fic_building = rio.open(args.building)
    mask_building = mask_fic_building.read(1)

    mask_fic_shadow = rio.open(args.shadow)
    mask_shadow = mask_fic_shadow.read(1)

    mask_fic_road = rio.open(args.road)
    mask_road = mask_fic_road.read(1)
    
    mnh_data = np.zeros_like(mask_veg)
    height_threshold = 0
    # by default, MNH_data is set to 0 and heigth threshold is set to 0m, 
    # so "mnh_data >= heigth_threshold (or <=) is always true.
    if args.mnh != "":
        mnh_fic = rio.open(args.mnh)
        mnh_data = mnh_fic.read(1)
        height_threshold = 5

    stack = np.zeros_like(mask_veg)
    height = np.zeros_like(mask_veg)
    confidence = np.zeros_like(mask_veg)

    # Urban layer : may be high, or "not confident"
    no_water_no_veg = np.logical_and(mask_water_pred == 0, mask_veg < 21)
    buildings_clean = np.logical_and(mask_building == 1, mask_road == 0)
    roads_clean = np.logical_and(mask_building == 0, mask_road == 1)


    buildings =   np.logical_and(np.logical_and(buildings_clean, mnh_data >= height_threshold), no_water_no_veg)
    roads = np.logical_and(roads_clean, no_water_no_veg)

    artificial = np.logical_or(mask_road == 1, mask_building == 1)
    
    urban_areas = np.logical_and(np.logical_and(artificial, mnh_data <= height_threshold), no_water_no_veg)

    stack[urban_areas] = UNDEF_URBAN_BARE_GROUND
    stack[buildings] = BUILDINGS
    stack[roads] = ROADS
    #stack[undefined_urban] = UNDEF_URBAN_BARE_GROUND

    height[buildings] = HIGH
    height[roads] = LOW
    height[urban_areas] = LOW

    #height[undefined_urban] = NOT_CONFIDENT

    confidence[buildings] += 1
    confidence[roads] += 1 

    bare_ground = np.logical_and(mask_water_pred == 0, mask_veg == 11)
    natural_bare_ground = np.logical_and(np.logical_and(mnh_data < height_threshold, artificial == False), bare_ground)
    
    stack[natural_bare_ground] = BARE_GROUND
    height[natural_bare_ground] = LOW

    confidence[natural_bare_ground] += 1
    
    low_veg = mask_veg == 21
    high_veg = np.logical_or(mask_veg == 22, mask_veg == 23)
    if args.use_mnh_veg == True:
        low_veg = np.logical_and(low_veg, mnh_data <= height_threshold)
        high_veg = np.logical_and(high_veg, mnh_data >= height_threshold)

    stack[low_veg] = LOW_VEG
    stack[high_veg] = HIGH_VEG

    height[low_veg] = LOW
    height[high_veg] = HIGH    

    confidence[high_veg] += 1
    confidence[low_veg] += 1

    # Water layer : may be low (classif.tif) or not confident (predict.tif + urban)
    water = mask_water == 1
    water_pred = np.logical_and(mask_veg < 11, np.logical_and(mask_water_pred == 1, mask_building == 0))
    mix_water_building = np.logical_and(mask_veg < 11, np.logical_and(mask_water_pred == 1, mask_building == 1))

    #stack[mix_water_building] = UNDEF_WATER_URBAN
    confidence[mix_water_building] += 1
    stack[water_pred] = WATER_PRED
    stack[water] = WATER
 
    height[water_pred] = LOW
    height[mix_water_building] = NOT_CONFIDENT
    height[water] = LOW

    confidence[water] += 1
    confidence[water_pred] += 1
   
    # Shadow
    shadow_pred = mask_shadow == 1
    #stack[shadow_pred] = SHADOW
    height[shadow_pred] = NOT_CONFIDENT

    confidence[shadow_pred] = -1



    # nb of layers
    profile['count'] = 3
    profile.update({'compress' : 'lzw'})
    with rio.open(args.mask, 'w+', **profile) as stack_file:
        stack_file.write(stack, 1)
        stack_file.write(height, 2)
        stack_file.write(confidence, 3)
        #Add class_label in the metadata
        stack_file.update_tags(class_label=['undef','low_vegetation','high_vegetation', 'water', 'buildings', 'undef_water_urban', 'bare_ground', 'undef_urban_bare_ground', 'water_pred', 'shadow', 'roads'])
    return


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("mask", help="Final mask")
        parser.add_argument("-vegetation", default=None, help="Vegetation mask")
        parser.add_argument("-water", default=None, help="Water mask (filtered mask)")
        parser.add_argument("-water_pred", default=None, help="Water mask (prediction)")
        parser.add_argument("-building", default=None, help="Building mask")
        parser.add_argument("-road", default=None, help="Road mask")
        parser.add_argument("-mnh", default="", help="Height elevation model")
        parser.add_argument("-shadow", default=None, help="Shadow mask")
        parser.add_argument("-use_mnh_veg", default=False, help="Use MNH to help categorize low/high veg")
        args = parser.parse_args()
        stack(args)

    except FileNotFoundError as fnfe_exception:
        print("FileNotFoundError", fnfe_exception)

    except PermissionError as pe_exception:
        print("PermissionError", pe_exception)

    except ArithmeticError as ae_exception:
        print("ArithmeticError", ae_exception)

    except MemoryError as me_exception:
        print("MemoryError", me_exception)

    except Exception as exception:  # pylint: disable=broad-except
        print("oups...", exception)
        traceback.print_exc()


if __name__ == "__main__":
    main()

    
