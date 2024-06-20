#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Compute water mask of PHR image with help of Pekel and Hand images."""


import argparse
import gc
import time
import traceback
from os.path import dirname, join
from subprocess import call
import json

import numpy as np
import otbApplication as otb
import rasterio as rio
from skimage.filters.rank import maximum
from skimage.measure import label, regionprops
from skimage.morphology import (
    area_closing,
    binary_closing,
    remove_small_holes,
    square, disk
)
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz, export_text
import concurrent.futures
from multiprocessing import shared_memory, get_context
from pylab import *
import uuid
from slurp.tools import io_utils

from slurp.tools import eoscale_utils
import eoscale.manager as eom
import eoscale.eo_executors as eoexe

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ModuleNotFoundError:
    print("Intel(R) Extension/Optimization for scikit-learn not found.")  


def superimpose(file_in, file_ref, file_out, type_out, write=False):
    """SuperImpose file_in with file_ref, output to file_out."""

    start_time = time.time()
    app = otb.Registry.CreateApplication("Superimpose")
    app.SetParameterString("inm", file_in)  # pekel or hand vrt
    app.SetParameterString("inr", file_ref)  # phr file
    app.SetParameterString("interpolator", "nn")
    app.SetParameterString("out", file_out + "?&writerpctags=true")
    app.SetParameterOutputImagePixelType("out", type_out)
    app.Execute()
    
    res = np.int16(np.copy(app.GetVectorImageAsNumpyArray("out")))
    
    if write:
        app.WriteOutput()
       
    print("Superimpose in", time.time() - start_time, "seconds.")
    
    return res


def pekel_recovery(file_ref, file_out, write=False):
    """Recover Occurrence Pekel image."""
    
    if write:
        print("Recover Occurrence Pekel file to", file_out)
    else:
        print("Recover Occurrence Pekel file")    
    pekel_image = superimpose(
        "/work/datalake/static_aux/MASQUES/PEKEL/data2021/occurrence/occurrence.vrt",
        file_ref,
        file_out,
        otb.ImagePixelType_uint8,
        write
    )
    
    return pekel_image.transpose(2,0,1)[0]


def pekel_month_recovery(file_ref, month, file_data_out, file_mask_out, write=False):
    """Recover Monthly Recurrence Pekel image.
    monthlyRecurrence and has_observations are signed int8 but coded on int16.
    """

    if write:
        print("Recover Monthly Recurrence Pekel file to", file_data_out)
    else:
        print("Recover Monthly Recurrence Pekel file") 
        
    pekel_image = superimpose(
        "/work/datalake/static_aux/MASQUES/PEKEL/data2021/MonthlyRecurrence/"
        f"monthlyRecurrence{month}/monthlyRecurrence{month}.vrt",
        file_ref,
        file_data_out,
        otb.ImagePixelType_int16,
        write
    )

    pekel_mask_out = superimpose(
        "/work/datalake/static_aux/MASQUES/PEKEL/data2021/MonthlyRecurrence/"
        f"has_observations{month}/has_observations{month}.vrt",
        file_ref,
        file_mask_out,
        otb.ImagePixelType_int16,
        write
    )
    
    return pekel_image.transpose(2,0,1)[0]


def hand_recovery(file_ref, file_out, write=False):
    """Recover HAND image."""

    if write:
        print("Recover HAND file to", file_out)
    else:
        print("Recover HAND file")        
    hand_image = superimpose(
        "/work/datalake/static_aux/MASQUES/HAND_MERIT/" "hnd.vrt",
        file_ref,
        file_out,
        otb.ImagePixelType_float,
        write
    )
    
    return hand_image.transpose(2,0,1)[0]


def esri_recovery(file_ref, file_out):
    """Recover ESRI image."""

    print("TODO: Recover ESRI file to", file_out)


def cloud_from_gml(file_cloud, file_ref):
    """Compute cloud mask from GML file."""

    start_time = time.time()
    app = otb.Registry.CreateApplication("Rasterization")
    app.SetParameterString("in", file_cloud)
    app.SetParameterString("im", file_ref)
    app.SetParameterFloat("background", 0)
    app.SetParameterString("mode", "binary")
    app.SetParameterFloat("mode.binary.foreground", 1)
    app.Execute()

    mask_cloud = app.GetImageAsNumpyArray(
        "out", otb.ImagePixelType_uint8
    ).astype(np.uint8)
    print("Rasterize clouds in", time.time() - start_time, "seconds.")

    return mask_cloud


def compute_mask(im_ref, im_nodata, thresh_ref):
    """Compute mask with one or multiple threshold values."""
    
    valid_ref = im_ref != im_nodata
    if isinstance(thresh_ref, list):
        mask_ref = []
        for thresh in thresh_ref:
            mask_ref.append(im_ref > thresh)
    else:
        mask_ref = im_ref > thresh_ref    
    del im_ref
    
    return mask_ref, valid_ref

def compute_mask_threshold(input_buffers: list, 
                  input_profiles: list, 
                  params: dict) -> np.ndarray :
    
    """ 
    Simple threshold on input image
    """
    res = np.zeros(input_buffers[0][0].shape)
    res = np.where(input_buffers[0][0] > params["threshold"], 1, 0)
    res = np.where(input_buffers[1][0] != 1, 255, res)

    return res


def compute_ndwi(input_buffers: list, 
                  input_profiles: list, 
                  params: dict) -> np.ndarray :
    """Compute Normalize Difference X Index.
    Rescale to [-1000, 1000] int16 with nodata value = 32767
    1000 * (im_b1 - im_b2) / (im_b1 + im_b2)
    """

    np.seterr(divide="ignore", invalid="ignore")

    im_ndwi = 1000.0 - (2000.0 * np.float32(input_buffers[0][params.nir-1])) / (
        np.float32(input_buffers[0][params.green-1]) + np.float32(input_buffers[0][params.nir-1]))
    im_ndwi[np.logical_or(im_ndwi < -1000.0, im_ndwi > 1000.0)] = np.nan
    np.nan_to_num(im_ndwi, copy=False, nan=32767)
    im_ndwi = np.int16(im_ndwi)

    return im_ndwi


def compute_ndvi(input_buffers: list, 
                  input_profiles: list, 
                  params: dict) -> np.ndarray :
    """Compute Normalize Difference X Index.
    Rescale to [-1000, 1000] int16 with nodata value = 32767
    1000 * (im_b1 - im_b2) / (im_b1 + im_b2)
    """

    np.seterr(divide="ignore", invalid="ignore")

    im_ndvi = 1000.0 - (2000.0 * np.float32(input_buffers[0][params.red-1])) / (
        np.float32(input_buffers[0][params.nir-1]) + np.float32(input_buffers[0][params.red-1]))
    im_ndvi[np.logical_or(im_ndvi < -1000.0, im_ndvi > 1000.0)] = np.nan
    np.nan_to_num(im_ndvi, copy=False, nan=32767)
    im_ndvi = np.int16(im_ndvi)
    
    return im_ndvi


def compute_valid_stack(inputBuffer: list, 
            input_profiles: list, 
            args: dict) -> list:
    #inputBuffer = [im_phr, mask_nocloud]
    # Valid_phr (boolean numpy array, True = valid data, False = no data)
    valid_phr = np.logical_and.reduce(inputBuffer[0] != args.nodata_phr, axis=0)
    valid_stack_cloud = np.logical_and(valid_phr, inputBuffer[1])
    
    return valid_stack_cloud
    

def compute_pekel_mask (inputBuffer: list, 
            input_profiles: list, 
            args: dict) -> list:
    """Compute Pekel mask regarding entry arguments."""
    
    if args.hand_strict:
        if not args.no_pekel_filter:
            [mask_pekel, mask_pekelxx, mask_pekel0] = compute_mask(inputBuffer[0], args.pekel_nodata, [args.thresh_pekel, args.strict_thresh, 0])[0]
        else:
            [mask_pekel, mask_pekelxx] = compute_mask(inputBuffer[0], args.pekel_nodata, [args.thresh_pekel, args.strict_thresh])[0]
            mask_pekel0 = np.zeros(inputBuffer[0].shape)
        return mask_pekel, mask_pekelxx
    
    elif not args.no_pekel_filter:
        [mask_pekel, mask_pekel0] = compute_mask(inputBuffer[0], args.pekel_nodata, [args.thresh_pekel, 0])[0]
    else:
        mask_pekel = compute_mask(inputBuffer[0], args.pekel_nodata, args.thresh_pekel)[0]
        mask_pekel0 = np.zeros(inputBuffer[0].shape)
    
    return [mask_pekel, mask_pekel0]


def compute_hand_mask(inputBuffer: list, 
            input_profiles: list, 
            args: dict) -> list:
    """Compute Hand mask with one or multiple threshold values."""
    
    valid_ref = inputBuffer[0] != args.hand_nodata
    mask_hand = inputBuffer[0] > args.thresh_hand    
    
    # Do not learn in water surface (usefull if image contains big water surfaces)
    # Add some robustness if hand_strict is not used
    #if args.hand_strict:
        #np.logical_not(np.logical_or(mask_hand, inputBuffer[1]), out=mask_hand)
    #else:
        #np.logical_not(mask_hand, out=mask_hand)
    np.logical_not(mask_hand, out=mask_hand)
    
    return  mask_hand      #[mask_hand, valid_ref]


def compute_filter(file_filter, desc_filter):
    """Compute filter mask. Water value is 1."""

    id_water = 1

    ds_filter = rio.open(file_filter)
    im_filter = ds_filter.read(1)
    valid_filter = im_filter != ds_filter.nodata
    mask_filter = im_filter == id_water
    io_utils.print_dataset_infos(ds_filter, desc_filter)
    ds_filter.close()
    del im_filter, ds_filter

    return mask_filter, valid_filter

def post_process(inputBuffer: list, 
            input_profiles: list, 
            args: dict) -> list :
    """Compute some filters on the prediction image."""
    # Inputs = [im_predict,mask_hand, mask_pekel0,valid_stack + file filters*x]
    buffer_shape=inputBuffer[0].shape
    
    # Filter with Hand
    if args.hand_filter:
        if not args.hand_strict:
            inputBuffer[0][np.logical_not(inputBuffer[1])] = 0
        else:
            print("\nWARNING: hand_filter and hand_strict are incompatible.")
                              
    # Filter for final classification
    if len(inputBuffer) > 4 or not args.no_pekel_filter:
        mask = np.zeros(buffer_shape, dtype=bool)
        if not args.no_pekel_filter:  # filter with pekel0
            mask = np.zeros(buffer_shape, dtype=bool)
            mask = np.logical_or(mask, inputBuffer[2][0])  # problème de mask_pekel0 if "not defined"
        for i in range(len(inputBuffer)-4):  # Other classification files
            filter_mask = compute_filter(inputBuffer[i+4], "FILTER " + str(i))[0]
            mask = np.logical_or(mask, filter_mask) 
 
        im_classif = mask_filter(inputBuffer[0], mask)
    else:
        im_classif = inputBuffer[0]
        
    # Closing
    if args.binary_closing:
        im_classif[0,:,:] = binary_closing(im_classif[0,:,:].astype(bool), disk(args.binary_closing)).astype(np.uint8)
    elif args.area_closing:
        im_classif[0,:,:] = area_closing(im_classif[0,:,:], args.area_closing, connectivity=2)
    elif args.remove_small_holes:
        im_classif[0,:,:] = remove_small_holes(
            im_classif[0,:,:].astype(bool), args.remove_small_holes, connectivity=2
        ).astype(np.uint8)
        
    # Add nodata in im_classif 
    im_classif[np.logical_not(inputBuffer[3])] = 255
    im_classif[im_classif == 1] = args.value_classif
    
    im_predict= inputBuffer[0]
    im_predict[np.logical_not(inputBuffer[3])] = 255
    im_predict[im_predict == 1] = args.value_classif
  
    return [im_predict, im_classif]


def get_random_indexes_from_masks(nb_indexes, mask_1, mask_2):
    """Get random valid indexes from masks.
    Mask 1 is a validity mask
    """
    rows_idxs = []
    cols_idxs = []
    
    if nb_indexes != 0 :    
        nb_idxs = 0

        height = mask_1.shape[0]
        width = mask_1.shape[1]

        nz_rows, nz_cols = np.nonzero(mask_2)
        min_row = np.min(nz_rows)
        max_row = np.max(nz_rows)
        min_col = np.min(nz_cols)
        max_col = np.max(nz_cols)

        while nb_idxs < nb_indexes:
            np.random.seed(712) # reproductible results
            row = np.random.randint(0, height)
            col = np.random.randint(0, width)

            if mask_1[row, col] and mask_2[row, col]:
                rows_idxs.append(row)
                cols_idxs.append(col)
                nb_idxs += 1
    
    return rows_idxs, cols_idxs

def get_grid_indexes_from_mask(nb_samples, valid_mask, mask_ground_truth):
    
    valid_samples = np.logical_and(mask_ground_truth, valid_mask).astype(np.uint8)
    _, rows, cols = np.where(valid_samples)

    if len(rows) >= nb_samples and nb_samples >= 1:
        # np.arange(0, len(rows) -1, ...) : to be sure to exclude index len(rows)
        # because in some cases (ex : 19871, 104 samples), last index is the len(rows)
        indices = np.arange(0, len(rows)-1, len(rows)/nb_samples).astype(np.uint16)

        s_rows = rows[indices]
        s_cols = cols[indices]
    else:
        s_rows = []
        s_cols = []
        
    return s_rows, s_cols
    

def get_smart_indexes_from_mask(nb_indexes, pct_area, minimum, mask):

    rows_idxs = []
    cols_idxs = []

    if nb_indexes != 0 :
        img_labels, nb_labels = label(mask, return_num=True)
        props = regionprops(img_labels)
        mask_area = float(np.sum(mask))

        # number of samples for each label/prop
        n1_indexes = int((1.0 - pct_area / 100.0) * nb_indexes / nb_labels)

        # number of samples to distribute to each label/prop
        n2_indexes = pct_area / 100.0 * nb_indexes / mask_area

        for prop in props:
            n3_indexes = n1_indexes + int(n2_indexes * prop.area)
            n3_indexes = max(minimum, n3_indexes)

            min_row = np.min(prop.bbox[0])
            max_row = np.max(prop.bbox[2])
            min_col = np.min(prop.bbox[1])
            max_col = np.max(prop.bbox[3])

            nb_idxs = 0
            while nb_idxs < n3_indexes:
                np.random.seed(712) # reproductible results
                row = np.random.randint(min_row, max_row)
                col = np.random.randint(min_col, max_col)

                if mask[row, col]:
                    rows_idxs.append(row)
                    cols_idxs.append(col)
                    nb_idxs += 1

    return rows_idxs, cols_idxs


def save_indexes(
    filename, water_idxs, other_idxs, shape, crs, transform, rpc, colormap
):
    """Save points used for learning into a file."""

    img = np.zeros(shape, dtype=np.uint8)

    for row, col in water_idxs:
        img[row, col] = 1

    for row, col in other_idxs:
        img[row, col] = 2

    img_dilat = maximum(img, square(5))
    io_utils.save_image(img_dilat, filename, crs, transform, 0, rpc, colormap)

    return


def mask_filter(im_in, mask_ref):
    """Remove water areas in im_in not in contact
    with water areas in mask_ref.
    """

    im_label, nb_label = label(im_in, connectivity=2, return_num=True)

    start_time = time.time()
    im_label_thresh = np.copy(im_label)
    im_label_thresh[np.logical_not(mask_ref)] = 0
    valid_labels = np.delete(np.unique(im_label_thresh), 0)

    im_filtered = np.zeros(np.shape(mask_ref), dtype=np.uint8)
    im_filtered[np.isin(im_label, valid_labels)] = 1

    return im_filtered


def print_feature_importance(classifier, feature_names):
    """Compute feature importance."""

    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]

    std = np.std(
        [tree.feature_importances_ for tree in classifier.estimators_], axis=0
    )

    print("Feature ranking:")
    for idx in indices:
        print(
            "  %4s (%f) (std=%f)"
            % (feature_names[idx], importances[idx], std[idx])
        )

def concatenate_samples(output_scalars, chunk_output_scalars, tile):
    
    output_scalars.append(chunk_output_scalars[0])
   
        
def build_samples(inputBuffer: list, 
                    input_profiles: list, 
                    args: dict) -> list:
    """Build samples."""
    # inputBuffer :[ mask_pekel,valid_stack, mask_hand, im_phr, im_ndvi, im_ndwi]
    # Retrieve number of pixels for each class
    
    nb_valid_subset = np.count_nonzero(inputBuffer[1])

    valid_water_pixels = np.logical_and(inputBuffer[0], inputBuffer[5] > args.ndwi_threshold)
    
    nb_water_subset = np.count_nonzero(np.logical_and(valid_water_pixels, inputBuffer[1]))
    
    nb_other_subset = nb_valid_subset - nb_water_subset
    # Ratio of pixel class compare to the full image ratio
    water_ratio = nb_water_subset/args.nb_valid_water_pixels
    other_ratio = nb_other_subset/args.nb_valid_other_pixels
    # Retrieve number of samples to create for each class in this subset 
    nb_water_subsamples = round(water_ratio*args.nb_samples_water)
    nb_other_subsamples = round(other_ratio*args.nb_samples_other) 

    # Prepare random water and other samples
    if args.nb_samples_auto:
        nb_water_subsamples = int(nb_water_subset * args.auto_pct)
        nb_other_subsamples = int(nb_other_subset * args.auto_pct)
            
    # Pekel samples
    if args.samples_method == "random":
        rows_pekel, cols_pekel = get_random_indexes_from_masks(
            nb_water_subsamples, inputBuffer[1][0], inputBuffer[0][0]
        )
        # Hand samples, always random (currently)
        rows_hand, cols_hand = get_random_indexes_from_masks(
            nb_other_subsamples, inputBuffer[1][0], inputBuffer[2][0]
        )

    if args.samples_method == "smart":
        rows_pekel, cols_pekel = get_smart_indexes_from_mask(
            nb_water_subsamples,
            args.smart_area_pct,
            args.smart_minimum,
            np.logical_and(inputBuffer[0][0], inputBuffer[1][0]),
        )
        # Hand samples, always random (currently)
        rows_hand, cols_hand = get_random_indexes_from_masks(
            nb_other_subsamples, inputBuffer[1][0], inputBuffer[2][0]
        )
            
            
    if args.samples_method == "grid":
        rows_pekel, cols_pekel = get_grid_indexes_from_mask(nb_water_subsamples,
                                                            inputBuffer[1],
                                                            valid_water_pixels[0])
        
        # Hand samples, always random (currently)
        rows_hand, cols_hand = get_grid_indexes_from_mask(nb_other_subsamples,
                                                              inputBuffer[1],
                                                              inputBuffer[2])


    # All samples
    rows = np.concatenate((rows_pekel, rows_hand))
    cols = np.concatenate((cols_pekel, cols_hand))
    if args.save_mode == "debug":
        colormap = {
            0: (0, 0, 0, 0),  # nodata
            1: (0, 0, 255),  # eau
            2: (255, 0, 0),  # autre
            3: (0, 0, 0, 0),
        }
        save_indexes(
            "samples.tif",
            zip(rows_pekel, cols_pekel),
            zip(rows_hand, cols_hand),
            args.shape,
            args.crs,
            args.transform,
            args.rpc,
            colormap,
        )


    # Prepare samples for learning
    im_stack = np.concatenate((inputBuffer[3],inputBuffer[4],inputBuffer[5],inputBuffer[0]),axis=0)
    samples = np.transpose(im_stack[:, rows.astype(np.uint16), cols.astype(np.uint16)])

    return samples   #[x_samples, y_samples]


def train_classifier(classifier, x_samples, y_samples):
    """Create and train classifier on samples."""
    
    start_time = time.time()
    x_train, x_test, y_train, y_test = train_test_split(
        x_samples, y_samples, test_size=0.2, random_state=42
    )
    
    classifier.fit(x_train, y_train)
    print("Train time :", time.time() - start_time)
    
    # Compute accuracy on train and test sets
    print(
        "Accuracy on train set :",
        accuracy_score(y_train, classifier.predict(x_train)),
    )
    print(
        "Accuracy on test set :",
        accuracy_score(y_test, classifier.predict(x_test)),
    )


def RF_prediction(inputBuffer: list, 
            input_profiles: list, 
            params: dict) -> list:
    
    #inputBuffer = [key_phr, key_ndvi[0], key_ndwi[0], valid_stack_key[0]]
    im_stack = np.concatenate((inputBuffer[0],inputBuffer[1],inputBuffer[2]),axis=0)
    buffer_to_predict=np.transpose(im_stack[:,inputBuffer[3][0]])

    classifier =params["classifier"]
    prediction = np.zeros(inputBuffer[3][0].shape, dtype=np.uint8)
    if buffer_to_predict.shape[0] == 0:
        print(f"WARNING > zone with NO DATA")
    else:
        prediction[inputBuffer[3][0]] = classifier.predict(buffer_to_predict)  
    
    return prediction
    

def convert_time(seconds):
    full_time = time.gmtime(seconds)
    return time.strftime("%H:%M:%S", full_time)


def getarguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Compute Water Mask.")

    group1 = parser.add_argument_group(description="*** INPUT FILES ***")
    group2 = parser.add_argument_group(description="*** OPTIONS ***")
    group3 = parser.add_argument_group(
        description="*** LEARNING SAMPLES SELECTION AND CLASSIFIER ***")
    group4 = parser.add_argument_group(description="*** POST PROCESSING ***")
    group5 = parser.add_argument_group(description="*** OUTPUT FILE ***")
    group6 = parser.add_argument_group(description="*** PARALLEL COMPUTING ***")

    # Input files
    group1.add_argument("main_config", help="First JSON file, load basis arguments")
    group1.add_argument("-user_config", help="Second JSON file, overload basis arguments if keys are the same")
    group1.add_argument("-file_vhr", help="PHR filename")

    group1.add_argument(
        "-pekel",
        required=False,
        action="store",
        dest="extracted_pekel",
        help="Pekel filename (computed if missing option)",
    )

    group1.add_argument(
        "-thresh_pekel",
        type=float,
        required=False,
        action="store",
        dest="thresh_pekel",
        help="Pekel Threshold float (default is 50)",
    )

    group1.add_argument(
        "-pekel_month",
        type=int,
        required=False,
        action="store",
        dest="pekel_month",
        help="Use monthly recurrence map instead of occurence map",
    )

    group1.add_argument(
        "-hand",
        required=False,
        action="store",
        dest="file_hand",
        help="Hand filename (computed if missing option)",
    )

    group1.add_argument(
        "-thresh_hand",
        type=int,
        required=False,
        action="store",
        dest="thresh_hand",
        help="Hand Threshold int >= 0 (default is 25)",
    )

    group1.add_argument(
        "-ndvi",
        required=False,
        action="store",
        dest="file_ndvi",
        help="NDVI filename (computed if missing option)",
    )

    group1.add_argument(
        "-ndwi",
        required=False,
        action="store",
        dest="file_ndwi",
        help="NDWI filename (computed if missing option)",
    )

    group1.add_argument(
        "-layers",
        nargs="+",
        required=False,
        action="store",
        dest="files_layers",
        metavar="FILE_LAYER",
        help="Add layers as features used by learning algorithm",
    )

    group1.add_argument(
        "-cloud_gml",
        required=False,
        action="store",
        dest="file_cloud_gml",
        help="Cloud file in .GML format",
    )
    
    group1.add_argument(
        "-filters",
        nargs="+",
        required=False,
        action="store",
        dest="file_filters",
        help="Add files used in filtering (postprocessing)",
    )

    # Options
    group2.add_argument("-red", default=1, help="Red band index")
    group2.add_argument("-nir", default=4, help="NIR band index")
    group2.add_argument("-green", default=2, help="green band index")

    
    group2.add_argument(
        "-hand_strict",
        required=False,
        action="store_true",
        dest="hand_strict",
        help="Use not(pekelxx) for other (no water) samples",
    )

    group2.add_argument(
        "-strict_thresh",
        type=float,
        required=False,
        action="store",
        dest="strict_thresh",
        help="Pekel Threshold float (default is 50)",
    )
    
    group2.add_argument(
        "-save_mode",
        choices=["none", "prim", "aux", "all", "debug"],
        required=False,
        action="store",
        dest="save_mode",
        help="Save all files (debug), only primitives (prim), only pekel and hand (aux), primitives, pekel and hand (all) or only output mask (none)",
    )

    group2.add_argument(
        "-simple_ndwi_threshold",
        required = False,
        action = "store_true",
        dest="simple_ndwi_threshold",
        help="Compute water mask as a simple NDWI threshold - useful in arid places where no water is known by Peckel"
    )

    group2.add_argument(
        "-ndwi_threshold",
        required = False,
        type = float,
        action = "store",
        dest="ndwi_threshold",
        help="Threshold used when Pekel is empty in the area"
    )

    # Samples
    group3.add_argument(
        "-samples_method",
        choices=["smart", "grid", "random"],
        required=False,
        action="store",
        dest="samples_method",
        help="Select method for choosing learning samples",
    )

    group3.add_argument(
        "-nb_samples_water",
        type=int,
        required=False,
        action="store",
        dest="nb_samples_water",
        help="Number of samples in water for learning (default is 2000)",
    )

    group3.add_argument(
        "-nb_samples_other",
        type=int,
        required=False,
        action="store",
        dest="nb_samples_other",
        help="Number of samples in other for learning (default is 10000)",
    )

    group3.add_argument(
        "-nb_samples_auto",
        required=False,
        action="store_true",
        dest="nb_samples_auto",
        help="Auto select number of samples for water and other",
    )

    group3.add_argument(
        "-auto_pct",
        type=float,
        required=False,
        action="store",
        dest="auto_pct",
        help="Percentage of samples points, to use with -nb_samples_auto",
    )

    group3.add_argument(
        "-smart_area_pct",
        type=int,
        required=False,
        action="store",
        dest="smart_area_pct",
        help="For smart method, importance of area for selecting number of samples in each water surface.",
    )

    group3.add_argument(
        "-smart_minimum",
        type=int,
        required=False,
        action="store",
        dest="smart_minimum",
        help="For smart method, minimum number of samples in each water surface.",
    )

    group3.add_argument(
        "-grid_spacing",
        type=int,
        required=False,
        action="store",
        dest="grid_spacing",
        help="For grid method, select samples on a regular grid (40 pixels seems to be a good value)",
    )

    group3.add_argument(
        "-max_depth",
        type=int,
        required=False,
        action="store",
        dest="max_depth",
        help="Max depth of trees"
    )

    group3.add_argument(
        "-nb_estimators",
        type=int,
        required=False,
        action="store",
        dest="nb_estimators",
        help="Nb of trees in Random Forest"
    )

    group3.add_argument(
        "-n_jobs",
        type=int,
        required=False,
        action="store",
        dest="nb_jobs",
        help="Nb of parallel jobs for Random Forest (1 is recommanded : use n_workers to optimize parallel computing)"
    )
    
    # Post processing
    group4.add_argument(
        "-no_pekel_filter",
        required=False,
        action="store_true",
        dest="no_pekel_filter",
        help="Deactivate postprocess with pekel which only keeps surfaces already known by pekel",
    )

    group4.add_argument(
        "-hand_filter",
        default=False,
        required=False,
        action="store_true",
        dest="hand_filter",
        help="Postprocess with Hand (set to 0 when hand > thresh), incompatible with hand_strict",
    )

    group4.add_argument(
        "-binary_closing",
        type=int,
        required=False,
        action="store",
        dest="binary_closing",
        help="Size of square structuring element",
    )

    group4.add_argument(
        "-area_closing",
        type=int,
        required=False,
        action="store",
        dest="area_closing",
        help="Area closing removes all dark structures",
    )

    group4.add_argument(
        "-remove_small_holes",
        type=int,
        required=False,
        action="store",
        dest="remove_small_holes",
        help="The maximum area, in pixels, of a contiguous hole that will be filled",
    )

    # Output
    group5.add_argument("-watermask", help="Output classification filename")

    group5.add_argument(
        "-value_classif",
        type=int,
        required=False,
        action="store",
        dest="value_classif",
        help="Output classification value (default is 1)",
    )

    # Parallel computing
    group6.add_argument(
        "-max_mem",
        type=int,
        required=False,
        action="store",
        dest="max_memory",
        help="Max memory permitted for the prediction of the Random Forest (in Gb)"
    )
    
    group6.add_argument(
        "-n_workers",
        type=int,
        required=False,
        action="store",
        dest="n_workers",
        help="Nb of CPU"
    )

    return parser.parse_args()

 ################ Main function ################
    
def main():
   
    argparse_dict = vars(getarguments())
    # Get the input file path from the command line argument
    arg_file_path_1 = argparse_dict["main_config"]

    # Read the JSON data from the input file
    try:
        with open(arg_file_path_1, 'r') as json_file1:
            full_args=json.load(json_file1)
            argsdict = full_args['input']
            argsdict.update(full_args['aux_layers'])
            argsdict.update(full_args['masks'])
            argsdict.update(full_args['ressources'])
            argsdict.update(full_args['water'])
            
            # a effacer après migration du pre-processing:
            argsdict.update(full_args['pre_process'])

    except FileNotFoundError:
        print(f"File {arg_file_path} not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON data from {arg_file_path_1}. Please check the file format.")

    if argparse_dict["user_config"] :   
    # Get the input file path from the command line argument
        arg_file_path_2 = argparse_dict["user_config"]

        # Read the JSON data from the input file
        try:
            with open(arg_file_path_2, 'r') as json_file2:
                full_args=json.load(json_file2)
                for k in full_args.keys():
                    if k in ['input','aux_layers','masks','ressources', 'water']:
                        argsdict.update(full_args[k])

        except FileNotFoundError:
            print(f"File {arg_file_path} not found.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON data from {arg_file_path_2}. Please check the file format.")

    #Overload with manually passed arguments if not None
    for key in argparse_dict.keys():
        if argparse_dict[key] is not None :
            argsdict[key]=argparse_dict[key]

    print("JSON data loaded:")
    print(argsdict)
    args = argparse.Namespace(**argsdict)  
    
    with eom.EOContextManager(nb_workers = args.n_workers, tile_mode = True) as eoscale_manager:
       
        try:

            t0 = time.time()
            
            ################ Build stack with all layers #######
            
            # Band positions in PHR image
            if args.red == 1:
                names_stack = ["R", "G", "B", "NIR", "NDVI", "NDWI"]
            else:
                names_stack = ["B", "G", "R", "NIR", "NDVI", "NDWI"]
            
            # Image PHR (numpy array, 4 bands, band number is first dimension),
            ds_phr = rio.open(args.file_vhr)
            ds_phr_profile=ds_phr.profile
            io_utils.print_dataset_infos(ds_phr, "PHR")
            args.nodata_phr = ds_phr.nodata
           
            # Save crs, transform and rpc in args
            args.shape = ds_phr.shape
            args.crs = ds_phr.crs
            args.transform = ds_phr.transform
            args.rpc = ds_phr.tags(ns="RPC")

            ds_phr.close()
            del ds_phr
            
            # Store image in shared memmory
            key_phr = eoscale_manager.open_raster(raster_path = args.file_vhr)
            
            ### Compute NDVI 
            if not args.file_ndvi :
                key_ndvi = eoexe.n_images_to_m_images_filter(inputs = [key_phr],
                                                               image_filter = compute_ndvi,
                                                               filter_parameters=args,
                                                               generate_output_profiles = eoscale_utils.single_int16_profile,
                                                               stable_margin= 0,
                                                               context_manager = eoscale_manager,
                                                               multiproc_context= "fork",
                                                               filter_desc= "NDVI processing...")
                if (args.save_mode != "none" and args.save_mode != "aux"):
                    eoscale_manager.write(key = key_ndvi[0], img_path = args.watermask.replace(".tif","_NDVI.tif"))
            else:
                key_ndvi = [ eoscale_manager.open_raster(raster_path =args.file_ndvi) ]
                
            
            ### Compute NDWI        
            if not args.file_ndwi :
                key_ndwi = eoexe.n_images_to_m_images_filter(inputs = [key_phr],
                                                               image_filter = compute_ndwi,
                                                               filter_parameters=args,
                                                               generate_output_profiles = eoscale_utils.single_int16_profile,
                                                               stable_margin= 0,
                                                               context_manager = eoscale_manager,
                                                               multiproc_context= "fork",
                                                               filter_desc= "NDWI processing...")         
                if (args.save_mode != "none" and args.save_mode != "aux"):
                    eoscale_manager.write(key = key_ndwi[0], img_path = args.watermask.replace(".tif","_NDWI.tif"))
            else:
                key_ndwi= [ eoscale_manager.open_raster(raster_path =args.file_ndwi) ]

            
            # Get cloud mask if any
            if args.file_cloud_gml:
                cloud_mask_array = np.logical_not(
                    cloud_from_gml(args.file_cloud_gml, args.file_vhr)   
                )
                #save cloud mask
                io_utils.save_image(cloud_mask_array,
                    join(dirname(args.watermask), "nocloud.tif"),
                    args.crs,
                    args.transform,
                    None,
                    args.rpc,
                    tags=args.__dict__,
                )
                mask_nocloud_key = eoscale_manager.open_raster(raster_path = join(dirname(args.watermask), "nocloud.tif"))   
                
            else:
                # Get profile from im_phr
                profile = eoscale_manager.get_profile(key_phr)
                profile["count"] = 1
                profile["dtype"] = np.uint8
                mask_nocloud_key = eoscale_manager.create_image(profile)
                eoscale_manager.get_array(key=mask_nocloud_key).fill(1)

            # Global validity mask construction
            valid_stack_key = eoexe.n_images_to_m_images_filter(inputs = [key_phr, mask_nocloud_key],
                                                           image_filter = compute_valid_stack,   
                                                           filter_parameters=args,
                                                           generate_output_profiles = eoscale_utils.single_bool_profile,
                                                           stable_margin= 0,
                                                           context_manager = eoscale_manager,
                                                           multiproc_context= "fork",
                                                           filter_desc= "Valid stack processing...")            
            
            time_stack = time.time()
            
            ################ Build samples #################
            
            write = False if (args.save_mode == "none" or args.save_mode == "prim") else True
    
            #### Image Pekel recovery (numpy array, first band)
            if not args.extracted_pekel:
                if 1 <= args.pekel_month <= 12:
                    args.file_data_pekel = join(
                        dirname(args.watermask), f"pekel{args.pekel_month}.tif"
                    )
                    args.file_mask_pekel = join(
                        dirname(args.watermask),
                        f"has_observations{args.pekel_month}.tif",
                    )
                    args.extracted_pekel = args.file_data_pekel
                    im_pekel = pekel_month_recovery(
                        args.file_vhr,
                        args.pekel_month,
                        args.file_data_pekel,
                        args.file_mask_pekel,
                        write=True,
                    )
                else:
                    args.extracted_pekel = join(dirname(args.watermask), "pekel.tif")
                    im_pekel = pekel_recovery(args.file_vhr, args.extracted_pekel, write=True)   
                
                pekel_nodata = 255.0 
                
                
            ds_ref = rio.open(args.extracted_pekel)
            io_utils.print_dataset_infos(ds_ref, "PEKEL")
            pekel_nodata = ds_ref.nodata  # contradiction
            key_pekel=eoscale_manager.open_raster(raster_path =args.extracted_pekel)
            ds_ref.close()
            del ds_ref
        
            args.pekel_nodata=pekel_nodata
                
            ### Pekel valid masks 
            mask_pekel = eoexe.n_images_to_m_images_filter(inputs = [key_pekel] ,
                                                           image_filter = compute_pekel_mask,
                                                           filter_parameters=args,
                                                           generate_output_profiles = eoscale_utils.double_int_profile,
                                                           stable_margin= 0,
                                                           context_manager = eoscale_manager,
                                                           multiproc_context= "fork",
                                                           filter_desc= "Pekel valid mask processing...")

            
            # If user wants a simple threshold on NDWI values, we don't select samples and launch learning/prediction step
            select_samples = not(args.simple_ndwi_threshold)
            
            ### Check pekel mask
            # - if there are too few values : we threshold NDWI to detect water areas 
            # - if there are even no "supposed water areas" : stop machine learning process (flag select_samples=False)
            local_mask_pekel = eoscale_manager.get_array(mask_pekel[0])
            if np.count_nonzero(local_mask_pekel) < 2000:
                # In case they are too few Pekel pixels, we prefer to threshold NDWI and skip samples selection
                # Alternative would be to select samples in a thresholded NDWI..
                select_samples = False

            
            ### Image HAND (numpy array, first band)
            if not args.file_hand:
                args.file_hand = join(dirname(args.watermask), "hand.tif")
                im_hand = hand_recovery(args.file_vhr, args.file_hand, write=True)  
                hand_nodata = -9999.0    
                

            ds_hand = rio.open(args.file_hand)
            io_utils.print_dataset_infos(ds_hand, "HAND")
            hand_nodata = ds_hand.nodata
            key_hand = eoscale_manager.open_raster(raster_path =args.file_hand)
            ds_hand.close()
            del ds_hand    
                
            args.hand_nodata = hand_nodata 
                
            # Create HAND mask 
            mask_hand = eoexe.n_images_to_m_images_filter(inputs = [key_hand],  
                                            image_filter = compute_hand_mask,  #args.hand_strict impossible because of mask_pekel0 not sure
                                            filter_parameters=args,
                                            generate_output_profiles = eoscale_utils.single_float_profile,
                                            stable_margin= 0,
                                            context_manager = eoscale_manager,
                                            multiproc_context= "fork",
                                            filter_desc= "Hand valid mask processing...")   

            
            ################ Build samples ##################
            
            if select_samples == False:
                # Not enough supposed water areas : skip sample selection
                # --> we force NDWI threshold and deactivate Pekel filter,
                # otherwise it would remove afain supposed areas
                args.simple_ndwi_threshold = True
                args.no_pekel_filter = True
                print("Simple threshold mask NDWI > "+str(args.ndwi_threshold))
                key_predict = eoexe.n_images_to_m_images_filter(inputs = [key_ndwi[0], valid_stack_key[0]],
                                                               image_filter = compute_mask_threshold,
                                                               filter_parameters= {"threshold":1000*args.ndwi_threshold},
                                                                context_manager = eoscale_manager,
                                                                generate_output_profiles = eoscale_utils.single_float_profile,
                                                                multiproc_context= "fork",
                                                           filter_desc= "Simple NDWI threshold") 
                time_random_forest = time.time() 
                time_samples = time_random_forest           
            else:
                # Nominal case : select samples, train, predict
                #
                # Sample selection
                valid_stack= eoscale_manager.get_array(valid_stack_key[0])
                nb_valid_pixels = np.count_nonzero(valid_stack)
                args.nb_valid_water_pixels = np.count_nonzero(np.logical_and(local_mask_pekel, valid_stack))
                args.nb_valid_other_pixels = nb_valid_pixels - args.nb_valid_water_pixels
                
                samples = eoexe.n_images_to_m_scalars(inputs=[mask_pekel[0],valid_stack_key[0], mask_hand[0], key_phr, key_ndvi[0], key_ndwi[0]],   
                                                           image_filter = build_samples,   
                                                           filter_parameters = args,
                                                           nb_output_scalars = args.nb_samples_water+args.nb_samples_other,
                                                           context_manager = eoscale_manager,
                                                           concatenate_filter = concatenate_samples, 
                                                           output_scalars= [],
                                                           multiproc_context= "fork",
                                                           filter_desc= "Samples building processing...")       # samples=[x_samples, y_samples]

                
                time_samples = time.time()

                ################ Train classifier from samples ########

                classifier = RandomForestClassifier(
                n_estimators=args.nb_estimators, max_depth=args.max_depth, random_state=712, n_jobs=args.nb_jobs
                )
                print("RandomForest parameters:\n", classifier.get_params(), "\n")
                samples=np.concatenate(samples[:]) # A revoir si possible
                x_samples= samples[:,:-1]
                y_samples= samples[:,-1]
                train_classifier(classifier, x_samples, y_samples)
                print_feature_importance(classifier, names_stack)
                gc.collect()



                ######### Predict  ################
                key_predict= eoexe.n_images_to_m_images_filter(inputs = [key_phr, key_ndvi[0], key_ndwi[0],valid_stack_key[0]],       
                                                               image_filter = RF_prediction,
                                                               filter_parameters= {"classifier": classifier},
                                                               generate_output_profiles = eoscale_utils.single_float_profile,
                                                               stable_margin= 0,
                                                               context_manager = eoscale_manager,
                                                               multiproc_context= "fork",
                                                               filter_desc= "RF prediction processing...")             
                time_random_forest = time.time()
            
            ######### Post_processing  ################
            
            file_filters= [eoscale_manager.open_raster(raster_path =args.file_filters[i]) for i in range(len(args.file_filters))]
            
            im_classif = eoexe.n_images_to_m_images_filter(inputs = [key_predict[0],mask_hand[0],mask_pekel[1],valid_stack_key[0]] + file_filters,     
                                                           image_filter = post_process,
                                                           filter_parameters= args,
                                                           generate_output_profiles = eoscale_utils.double_int_profile,
                                                           stable_margin= 3,
                                                           context_manager = eoscale_manager,
                                                           multiproc_context= "fork",
                                                           filter_desc= "Post processing...")


            # Save predict and classif image
            final_predict = eoscale_manager.write(key=im_classif[0], img_path = join(dirname(args.watermask), "predict.tif"))
            final_classif = eoscale_manager.write(key=im_classif[1], img_path = args.watermask)

            end_time = time.time()

            print("**** Water mask for "+str(args.file_vhr)+" (saved as "+str(args.watermask)+") ****")
            print("Total time (user)       :\t"+convert_time(end_time-t0))
            print("- Build_stack           :\t"+convert_time(time_stack-t0))
            print("- Build_samples         :\t"+convert_time(time_samples-time_stack))
            print("- Random forest (total) :\t"+convert_time(time_random_forest-time_samples))
            print("- Post-processing       :\t"+convert_time(end_time-time_random_forest))
            print("***")
            print("Max workers used for parallel tasks "+str(args.n_workers))        
              
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
