#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script stacks existing masks
"""

import argparse
import rasterio
import numpy as np
import argparse
import os

def compute_mask(image, mask, threshold_RGB, threshold_NIR, percentile):
    ds_im= rasterio.open(image)
    profile = ds_im.profile
    profile.update({'compress' : 'lzw'})
    
    nir = ds_im.read(4)
    
    band = []
    th_band = []
    shadow_mask = np.zeros_like(nir)
    
    min_nir = np.percentile(nir, percentile)
    th_nir = threshold_NIR * (np.percentile(nir, 100-percentile) - min_nir)
        
    shadow = nir < min_nir+th_nir

    for cpt in range(1,4):
        band = ds_im.read(cpt)
        min_band = np.percentile(band,percentile)
        th_band  = threshold_RGB * (np.percentile(band, 100-percentile) - min_band)
        shadow = np.logical_and(shadow, band < min_band+th_band)            
    
    shadow_mask[shadow] = 1
    
    with rasterio.open(mask, 'w+', 
                       driver="GTiff",
                       compress="deflate",
                       height=ds_im.shape[0],
                       width=ds_im.shape[1],
                       crs=ds_im.crs,
                       transform=ds_im.transform,
                       count=1, dtype=np.uint8) as mask_file:
        mask_file.write(shadow_mask.astype(np.uint8),1)
    

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("image", help="Input 4 bands VHR image")
        parser.add_argument("mask", help="Final mask")
        parser.add_argument("-th_rgb", default=0.3, help="Relative shadow threshold for RGB bands")
        parser.add_argument("-th_nir", default=0.3, help="Relative shadow threshold for NIR band")
        parser.add_argument("-percentile", default=2, help="Percentile value to cut histogram and estimate shadow threshold")
        
        args = parser.parse_args()

        compute_mask(args.image, args.mask, float(args.th_rgb), float(args.th_nir), int(args.percentile))

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
    
    
