#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Compute building and road masks from VHR images thanks to OSM layers """

import argparse
import gc
import time
import traceback
from os.path import dirname, join, basename
from subprocess import call
from math import sqrt, ceil

import joblib
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from skimage.morphology import (
    area_closing,
    binary_closing,
    binary_opening,
    binary_dilation,
    diameter_closing,
    remove_small_holes,
    remove_small_objects,
    square, disk
)
from skimage import segmentation
from skimage.filters import rank
from skimage.measure import label
from skimage.filters import sobel
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
import random
from slum.tools import io_utils
import otbApplication as otb

from multiprocessing import shared_memory, get_context
import concurrent.futures
import sys
import uuid
import eoscale.manager as eom
import eoscale.eo_executors as eoexe

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ModuleNotFoundError:
    print("Intel(R) Extension/Optimization for scikit-learn not found.")

def single_float_profile(input_profiles: list, map_params):
    profile = input_profiles[0]
    profile['count']=1
    profile['dtype']=np.float32
    profile["compress"] = "lzw"
    
    return profile

def post_process_profile(input_profiles: list, map_params):
    profile = input_profiles[0]
    profile['count']=2
    profile['dtype']=np.uint8
    profile["compress"] = "lzw"
    profile["nodata"] = 255

    return profile

def single_int16_profile(input_profiles: list, map_params):
    profile= input_profiles[0]
    profile["count"]= 1
    profile["dtype"]= np.int16
    profile["nodata"] = 32767
    profile["compress"] = "lzw"
    
    return profile

def single_uint8_profile(input_profiles: list, map_params):
    profile = input_profiles[0]
    profile["count"]= 1
    profile["dtype"]= np.uint8
    profile["compress"] = "lzw"
    profile["nodata"] = 255
    
    return profile

def three_uint8_profile(input_profiles: list, map_params):
    profile = input_profiles[0]
    profile["count"]= 3
    profile["dtype"]= np.uint8
    profile["compress"] = "lzw"
    profile["nodata"] = 255
    
    return profile

def three_int16_profile(input_profiles: list, map_params):
    profile = input_profiles[0]
    profile["count"]= 3
    profile["dtype"]= np.int16
    profile["compress"] = "lzw"
    profile["nodata"] = 32767
    
    return profile

def single_bool_profile(input_profiles: list, map_params):
    profile = input_profiles[0]
    profile['count']=1
    profile['dtype']=bool
    profile["compress"] = "lzw"

    return profile
    
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
    im_ndwi[np.logical_not(input_buffers[1][0])] = np.nan
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
    im_ndvi[np.logical_not(input_buffers[1][0])] = np.nan
    np.nan_to_num(im_ndvi, copy=False, nan=32767)
    im_ndvi = np.int16(im_ndvi)
    
    return im_ndvi
    
    
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

def save_image(
        image,
        file,
        crs=None,
        transform=None,
        nodata=None,
        rpc=None,
        colormap=None,
        tags=None,
        dtype=None,
        **kwargs,
):
    """Save 1 band numpy image to file with deflate compression.
    Note that rio.dtype is string so convert np.dtype to string.
    rpc must be a dictionnary.
    """
    if dtype != None:
        type_save = dtype
    else:
        type_save = str(image.dtype)
    dataset = rio.open(
        file,
        "w",
        driver="GTiff",
        compress="deflate",
        height=image.shape[0],
        width=image.shape[1],
        count=1,
        dtype=type_save,
        crs=crs,
        transform=transform,
        **kwargs,
    )
    dataset.write(image, 1)
    dataset.nodata = nodata

    if rpc:
        dataset.update_tags(**rpc, ns="RPC")

    if colormap:
        dataset.write_colormap(1, colormap)

    if tags:
        dataset.update_tags(**tags)
        
    dataset.close()
    del dataset

def show_images(image1, title1, image2, title2, **kwargs):
    """Show 2 images with matplotlib."""

    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(14, 7), sharex=True, sharey=True
    )

    axes[0].imshow(image1, cmap=plt.gray(), **kwargs)
    axes[0].axis("off")
    axes[0].set_title(title1, fontsize=20)

    axes[1].imshow(image2, cmap=plt.gray(), **kwargs)
    axes[1].axis("off")
    axes[1].set_title(title2, fontsize=20)

    fig.tight_layout()
    plt.show()


def show_histograms(image1, title1, image2, title2, **kwargs):
    """Compute and show 2 histograms with matplotlib."""

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7), sharey=True)

    hist1, ignored = np.histogram(image1, bins=201, range=(-1000, 1000))
    hist2, ignored = np.histogram(image2, bins=201, range=(-1000, 1000))
    del ignored

    axes[0].plot(np.arange(-1000, 1001, step=10), hist1, **kwargs)
    axes[1].plot(np.arange(-1000, 1001, step=10), hist2, **kwargs)

    axes[0].set_title(title1)
    axes[1].set_title(title2)

    fig.tight_layout()
    plt.show()


def show_histograms2(image1, title1, image2, title2, **kwargs):
    """Compute and show 2 histograms with matplotlib."""

    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(14, 7))

    hist1, ignored = np.histogram(image1, bins=201, range=(-1000, 1000))
    hist2, ignored = np.histogram(image2, bins=201, range=(-1000, 1000))
    del ignored

    axe.plot(
        np.arange(-1000, 1001, step=10),
        hist1,
        color="blue",
        label=title1,
        **kwargs
    )
    axe.plot(
        np.arange(-1000, 1001, step=10),
        hist2,
        color="red",
        label=title2,
        **kwargs
    )

    fig.tight_layout()
    plt.legend()
    plt.show()


def show_histograms4(
    image1, title1, image2, title2, image3, title3, image4, title4, **kwargs
):
    """Compute and show 4 histograms with matplotlib."""

    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(14, 7))

    hist1, ignored = np.histogram(image1, bins=201, range=(-1000, 1000))
    hist2, ignored = np.histogram(image2, bins=201, range=(-1000, 1000))
    hist3, ignored = np.histogram(image3, bins=201, range=(-1000, 1000))
    hist4, ignored = np.histogram(image4, bins=201, range=(-1000, 1000))
    del ignored

    axe.plot(np.arange(-1000, 1001, step=10), hist1, label=title1, **kwargs)
    axe.plot(np.arange(-1000, 1001, step=10), hist2, label=title2, **kwargs)
    axe.plot(np.arange(-1000, 1001, step=10), hist3, label=title3, **kwargs)
    axe.plot(np.arange(-1000, 1001, step=10), hist4, label=title4, **kwargs)

    fig.tight_layout()
    plt.legend()
    plt.show()
    
    
def superimpose(file_in, file_ref, file_out, type_out, write=False):
    """SuperImpose file_in with file_ref, output to file_out."""

    start_time = time.time()
    app = otb.Registry.CreateApplication("Superimpose")
    app.SetParameterString("inm", file_in)  # wsf
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
    
    
def wsf_recovery(file_ref, file_out, write=False):
    """Recover WSF image."""

    if write:
        print("Recover WSF file to", file_out)
    else:
        print("Recover WSF file")        
    wsf_image = superimpose(
        "/work/datalake/static_aux/MASQUES/WSF/WSF2019_v1/WSF2019_v1.vrt",
        file_ref,
        file_out,
        otb.ImagePixelType_uint16,
        write
    )
    
    return wsf_image.transpose(2,0,1)[0]



def compute_mask(file_ref, field_value):
    """Compute mask with a threshold value."""

    ds_ref = rio.open(file_ref)
    im_ref = ds_ref.read(1)
    valid_ref = im_ref != ds_ref.nodata
    mask_ref = im_ref == field_value
    del im_ref, ds_ref

    return mask_ref, valid_ref


def compute_valid_stack(inputBuffer: list, 
            input_profiles: list, 
            args: dict) -> list:
    #inputBuffer = [im_phr, mask_nocloud]
    # Valid_phr (boolean numpy array, True = valid data, False = no data)
    valid_phr = np.logical_and.reduce(inputBuffer[0] != args.nodata_phr, axis=0)
    valid_stack_cloud = np.logical_and(valid_phr, inputBuffer[1])
    
    return valid_stack_cloud


def compute_esri(file_esri, desc_esri=""):
    """Compute ESRI mask. Water value is 1."""

    id_water = 1

    ds_esri = rio.open(file_esri)
    im_esri = ds_esri.read(1)
    valid_esri = im_esri != ds_esri.nodata
    mask_esri = im_esri == id_water
    print_dataset_infos(ds_esri, desc_esri)
    del im_esri, ds_esri

    return mask_esri, valid_esri


def get_crs_transform(file):
    """Get CRS annd Transform of a geotiff file."""

    dataset = rio.open(file)
    crs = dataset.crs
    transform = dataset.transform
    rpc = dataset.tags(ns="RPC")
    dataset.close()

    return crs, transform, rpc


def get_indexes_from_masks(nb_indexes, mask1, value1, mask_valid, args):
    """Get valid indexes from masks.
    Mask 1 is a validity mask
    """

    nb_idxs = 0
    number = args.random_seed
    rows_idxs = []
    cols_idxs = []

    height = mask_valid.shape[0]
    width = mask_valid.shape[1]

    if args.random_seed:
        np.random.seed(args.random_seed)
        row = np.random.randint(0, height)
        np.random.seed(args.random_seed + number)
        col = np.random.randint(0, height)

        while nb_idxs < nb_indexes:
            np.random.seed(row + number)
            row = np.random.randint(0, height)
            np.random.seed(col + number)
            col = np.random.randint(0, width)
            if mask1[row, col] == value1 and mask_valid[row, col]:
                rows_idxs.append(row)
                cols_idxs.append(col)
                nb_idxs += 1
            number += 1
    else:
        while nb_idxs < nb_indexes:
            row = np.random.randint(0, height)
            col = np.random.randint(0, width)
            if mask1[row, col] == value1 and mask_valid[row, col]:
                rows_idxs.append(row)
                cols_idxs.append(col)
                nb_idxs += 1
        # else:
        #    print("Row : "+str(row)+" Col : "+str(col)+" -> "+str(mask1[row, col]))
    return rows_idxs, cols_idxs



def save_indexes(filename, water_idxs, other_idxs, shape, crs, transform, rpc):
    img = np.zeros(shape, dtype=np.uint8)
    for row, col in water_idxs:
        img[row, col] = 1
    for row, col in other_idxs:
        img[row, col] = 2
    io_utils.save_image(img, filename, crs, transform, 0, rpc)
    return


def show_rftree(estimator, feature_names):
    """Display Random Forest Estimator."""

    export_graphviz(
        estimator,
        out_file="tree.dot",
        feature_names=feature_names,
        class_names=["Other", "Water"],
        rounded=True,
        proportion=False,
        precision=2,
        filled=True,
    )

    call(["dot", "-Tpng", "tree.dot", "-o", "tree.png", "-Gdpi=300"])
    print(
        export_text(estimator, show_weights=True, feature_names=feature_names)
    )


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
    #inputBuffer=[valid_stack_key[0],key_gt, key_phr, key_ndvi[0], key_ndwi[0]]+ files_layers 
    
    mask_building = inputBuffer[1]==args.value_classif
    
    # Retrieve number of pixels for each class
    nb_valid_subset = np.count_nonzero(inputBuffer[0])
    nb_built_subset = np.count_nonzero(np.logical_and(inputBuffer[1], inputBuffer[0]))
    nb_other_subset = nb_valid_subset - nb_built_subset
    # Ratio of pixel class compare to the full image ratio
    urban_ratio = nb_built_subset/ args.nb_valid_built_pixels
    other_ratio = nb_other_subset/args.nb_valid_other_pixels
    # Retrieve number of samples to create for each class in this subset 
    nb_urban_subsamples = round(urban_ratio*args.nb_samples_urban)
    nb_other_subsamples = round(other_ratio*args.nb_samples_other)

    # Building samples
    rows_b, cols_b = get_indexes_from_masks(
        nb_urban_subsamples, inputBuffer[1][0], args.value_classif, inputBuffer[0][0],args
    )
    
    rows_road = []
    cols_road = []

    if args.nb_classes == 2:
        rows_road, cols_road = get_indexes_from_masks(
            nb_urban_subsamples, inputBuffer[1][0], 2, inputBuffer[0][0], args
        )
    
    rows_nob = []
    cols_nob = []

    rows_nob, cols_nob = get_indexes_from_masks(
        nb_other_subsamples, inputBuffer[1][0], 0, inputBuffer[0][0], args
    )

    # samples = building+non building
    rows = rows_b + rows_nob
    cols = cols_b + cols_nob
    if args.nb_classes == 2:
        rows = rows + rows_road
        cols = cols + cols_road

    # Prepare samples for learning
#    im_stack = np.concatenate((np.concatenate(inputBuffer[1:-1], axis=0),mask_building),axis=0)  # TODO : gérer les files_layers optionnels
    im_stack = np.concatenate((inputBuffer[1:]), axis=0) # TODO : gérer les files_layers optionnels
    samples = np.transpose(im_stack[:, rows, cols])

    return samples


def train_classifier(classifier, x_samples, y_samples):
    """Create and train classifier on samples."""

    start_time = time.time()
    x_train, x_test, y_train, y_test = train_test_split(
        x_samples, y_samples, test_size=0.2, random_state=42
    )
    classifier.fit(x_train, y_train)
    print("Train time :", time.time() - start_time)

    # Compute accuracy on train and test sets
    x_train_prediction = classifier.predict(x_train)
    x_test_prediction = classifier.predict(x_test)
    
    print(
        "Accuracy on train set :",
        accuracy_score(y_train, x_train_prediction),
    )
    print(
        "Accuracy on test set :",
        accuracy_score(y_test, x_test_prediction),
    )


def get_bornes(index, step, limit, margin):
    if index == 0: # first band
        start_with_margin = 0
        start_extract = 0
    elif (index*step - margin) < 0:  # large margin       
        start_with_margin = 0
        start_extract = index*step
    else:
        start_with_margin = index*step - margin
        start_extract = margin
        
    if (index+1)*step > limit: # last band
        end_with_margin = limit
        end_extract = start_extract + limit - index*step
    else:
        end_with_margin = min((index+1)*step + margin, limit)
        end_extract = start_extract + step
    return start_with_margin, end_with_margin, start_extract, end_extract


def watershed_regul(args, clean_predict, inputBuffer):    
    # inputBuffer = [key_predict[0], key_phr, key_watermask[0], key_vegmask[0], gt_key, valid_stack_key[0]
    # Compute gradient : either on NDVI image, or on RGB image 
    # DEBUG im_stack = PHR image + ndvi + ndwi + file layers
    # Compute mono image from RGB image
    im_mono = 0.29*inputBuffer[1][0] + 0.58*inputBuffer[1][1] + 0.114*inputBuffer[1][2]
       
    # compute gradient
    edges = sobel(im_mono)

    del im_mono

    # markers map : -1, 1 and 2 : probable background, buildings or false positive
    # inputBuffer[0] = proba of building class
    markers = np.zeros_like(inputBuffer[0][0])
    probable_buildings = np.logical_and(inputBuffer[0][2] > args.confidence_threshold, clean_predict == 1)
    probable_background = np.logical_and(inputBuffer[0][2] < 20, clean_predict == 0)
    markers[probable_background] = 3
    markers[probable_buildings] = 1
    # vegetation
    markers[inputBuffer[3][0] > args.vegmask_max_value] = 3
    # water
    markers[inputBuffer[2][0] == 1] = 3
    del probable_buildings, probable_background        

    if args.remove_false_positive:
        ground_truth = inputBuffer[4][0]
        # mark as false positive pixels with high confidence but not covered by dilated ground truth
        false_positive = np.logical_and(binary_dilation(ground_truth, disk(20)) == 0, inputBuffer[0][2] > args.confidence_threshold)
        markers[false_positive] = 2
        del ground_truth, false_positive

    # watershed segmentation
    seg = segmentation.watershed(edges, markers)
    seg[seg==3] = 0
    
    # remove small artefacts 
    if args.remove_small_objects:
        res = remove_small_objects(seg.astype(bool), args.remove_small_objects, connectivity=2).astype(np.uint8)
        # res is either 0 or 1 : we multiply by seg to keep 0/1/2 classes
        seg = np.multiply(res, seg)

    return seg, markers, edges

def RF_prediction(inputBuffer: list, 
            input_profiles: list, 
            params: dict) -> list:
    """
    inputBuffer = [valid_stack_key[0], key_phr, key_ndvi[0], key_ndwi[0], + file_layers]
    buffer_to_predict are non NODATA pixels, defined by all the primitives (R-G-B-NIR-NDVI-NDWI-[+ features]
    
    """
    im_stack= np.concatenate((inputBuffer[1:]),axis=0)
    buffer_to_predict=np.transpose(im_stack[:,inputBuffer[0][0]])

    classifier =params["classifier"]
    proba = classifier.predict_proba(buffer_to_predict)

    # Prediction, inspired by sklearn code to predict class
    res_classif = classifier.classes_.take(np.argmax(proba, axis=1), axis=0)
    res_classif[res_classif == 255] = 1
    
    prediction = np.zeros((3,inputBuffer[0].shape[1],inputBuffer[0].shape[2]))
    # Class predicted
    prediction[0][inputBuffer[0][0]] = res_classif 
    # Proba for class 0 (background)
    prediction[1][inputBuffer[0][0]] = 100*proba[:,0]
    # Proba for class 1 (buildings)
    prediction[2][inputBuffer[0][0]] = 100*proba[:,1]
    
    return prediction

def post_process(inputBuffer: list, 
            input_profiles: list, 
            params: dict) -> list:
    # inputs = [key_predict[0],key_phr, key_watermas[0], key_vegmask[0],gt_key,valid_stack_key[0]
    # Clean
    im_classif = clean(params, inputBuffer[0][0])
    # Watershed regulation
    final_mask, markers, edges = watershed_regul(params, im_classif, inputBuffer)
    
    # Add nodata in final_mask (inputBuffer[5] : valid mask)
    final_mask[np.logical_not(inputBuffer[5][0])] = 255

    res_int = np.zeros((2,inputBuffer[1].shape[1],inputBuffer[1].shape[2]))
    res_int[0] = final_mask
    res_int[1] = markers

    return res_int

def convert_time(seconds):
    full_time = time.gmtime(seconds)
    return time.strftime("%H:%M:%S", full_time)


def clean(args, im_classif):
    #t0 = time.time()
    
    if args.binary_opening:
        # Opening can remove small bright spots (i.e. “salt”) and connect small dark cracks.
        im_classif = binary_opening(im_classif, square(args.binary_opening)).astype(np.uint8)

    if args.binary_closing:
        # Closing can remove small dark spots (i.e. “pepper”) and connect small bright cracks.
        im_classif = binary_closing(im_classif, square(args.binary_closing)).astype(np.uint8)

    if args.remove_small_objects:
        im_classif = remove_small_objects(
            im_classif.astype(bool), args.remove_small_objects, connectivity=2
        ).astype(np.uint8)


    return im_classif


def getarguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Compute Water Mask.")

    parser.add_argument("file_phr", help="PHR filename")

    parser.add_argument("-red", default=1, help="Red band index")
    parser.add_argument("-nir", default=4, help="NIR band index")
    parser.add_argument("-green", default=2, help="green band index")

    parser.add_argument(
        "-display",
        required=False,
        action="store_true",
        help="Display images while running",
    )
    
    parser.add_argument(
        "-save",
        choices=["none", "prim", "aux", "all", "debug"],
        default="none",
        required=False,
        action="store",
        dest="save_mode",
        help="Save all files (debug), only primitives (prim), only wsf (aux), primitives and wsf (all) or only output mask (none)",
    )

    parser.add_argument(
        "-ndvi",
        default=None,
        required=False,
        action="store",
        dest="file_ndvi",
        help="NDVI filename (computed if missing option)",
    )

    parser.add_argument(
        "-ndwi",
        default=None,
        required=False,
        action="store",
        dest="file_ndwi",
        help="NDWI filename (computed if missing option)",
    )
    
    parser.add_argument(
        "-watermask",
        default=None,
        required=False,
        action="store",
        dest="file_watermask",
        help="Watermask filename : urban mask will be learned & predicted, excepted on water areas"
    )
    
    parser.add_argument(
        "-vegetationmask",
        default=None,
        required=False,
        action="store",
        dest="file_vegetationmask",
        help="Vegetation mask filename : urban mask will be learned & predicted, excepted on vegetated areas"
    )

    parser.add_argument(
        "-vegmask_max_value",
        required=False,
        type=int,
        default=21,
        action="store",
        dest="vegmask_max_value",
        help="Vegetation mask value for vegetated areas : all pixels with lower value will be predicted"
    )

    parser.add_argument(
        "-cloud_gml",
        required=False,
        action="store",
        dest="file_cloud_gml",
        help="Cloud file in .GML format",
    )

    parser.add_argument(
        "-layers",
        nargs="+",
        default=[],
        required=False,
        action="store",
        dest="files_layers",
        metavar="FILE_LAYER",
        help="Add layers as features used by learning algorithm",
    )
    
    parser.add_argument(
        "-urban_raster",
        default=None,
        required=False,
        action="store",
        dest="urban_raster",
        help="Ground Truth (could be OSM, WSF). By default, WSF is automatically retrieved"
    )
    
    parser.add_argument(
        "-nb_classes",
        default=1,
        type=int,
        required=False,
        action="store",
        dest="nb_classes",
        help="Nb of classes in the ground-truth (1 by default - buildings only. Can be fix to 2 to classify buildings/roads"
    )

    parser.add_argument(
        "file_classif",
        help="Output classification filename (default is classif.tif)",
    )

    parser.add_argument(
        "-value_classif",
        type=int,
        default=255,
        required=False,
        action="store",
        dest="value_classif",
        help="Input ground truth class to consider in the input ground truth (default is 255 for WSF)",
    )

    parser.add_argument(
        "-nb_samples_urban",
        type=int,
        default=1000,
        required=False,
        action="store",
        dest="nb_samples_urban",
        help="Number of samples in buildings for learning (default is 1000)",
    )

    parser.add_argument(
        "-nb_samples_other",
        type=int,
        default=5000,
        required=False,
        action="store",
        dest="nb_samples_other",
        help="Number of samples in other for learning (default is 5000)",
    )

    parser.add_argument(
        "-max_depth",
        type=int,
        default=8,
        required=False,
        action="store",
        dest="max_depth",
        help="Max depth of trees"
    )

    parser.add_argument(
        "-nb_estimators",
        type=int,
        default=100,
        required=False,
        action="store",
        dest="nb_estimators",
        help="Nb of trees in Random Forest"
    )

    parser.add_argument(
        "-n_jobs",
        type=int,
        default=1,
        required=False,
        action="store",
        dest="nb_jobs",
        help="Nb of parallel jobs for Random Forest"
    )
    
    parser.add_argument(
        "-max_mem",
        type=int,
        default=25,
        required=False,
        action="store",
        dest="max_memory",
        help="Max memory permitted for the prediction of the Random Forest (in Gb)"
    )
    
    parser.add_argument(
        "-n_workers",
        type=int,
        default=8,
        required=False,
        action="store",
        dest="nb_workers",
        help="Nb of CPU"
    )
    
    parser.add_argument(
        "-margin",
        type=int,
        default=0,
        required=False,
        action="store",
        dest="margin",
        help="Margin for the regularization (clean and watershed)"
    )

    parser.add_argument(
        "-random_seed",
        type=int,
        default=712,
        required=False,
        action="store",
        dest="random_seed",
        help="Fix the random seed for samples selection",
    )

    parser.add_argument(
        "-binary_closing",
        type=int,
        default=0,
        required=False,
        action="store",
        dest="binary_closing",
        help="Size of square structuring element"
    )

    parser.add_argument(
        "-binary_opening",
        type=int,
        default=0,
        required=False,
        action="store",
        dest="binary_opening",
        help="Size of square structuring element"
    )
    
    parser.add_argument(
        "-remove_small_objects",
        type=int,
        default=0,
        required=False,
        action="store",
        dest="remove_small_objects",
        help="The minimum area, in pixels, of the objects to detect",
    )
    
    parser.add_argument(
        "-remove_false_positive",
        default=False,
        required=False,
        action="store_true",
        dest="remove_false_positive",
        help="Will dilate and use input ground-truth as mask to filter false positive from initial prediction"
    )
    
    parser.add_argument(
        "-confidence_threshold",
        type=int,
        default=85,
        required=False,
        action="store",
        dest="confidence_threshold",
        help="Confidence threshold to consider true positive in regularization step (85 by default)",
    )
    
    return parser.parse_args()



def main():
  
    args = getarguments()
    print(args)    
    with eom.EOContextManager(nb_workers = args.nb_workers, tile_mode = True) as eoscale_manager:
       
        try:
            
            t0 = time.time()
            
            ################ Build stack with all layers #######
            
            # Band positions in PHR image
            if args.red == 1:
                names_stack = ["R", "G", "B", "NIR", "NDVI", "NDWI"]
            else:
                names_stack = ["B", "G", "R", "NIR", "NDVI", "NDWI"]

            names_stack += [basename(f) for f in args.files_layers]
            
            # Image PHR (numpy array, 4 bands, band number is first dimension),
            ds_phr = rio.open(args.file_phr)
            ds_phr_profile=ds_phr.profile
            print_dataset_infos(ds_phr, "PHR")
            args.nodata_phr = ds_phr.nodata
           
            # Save crs, transform and rpc in args
            args.shape = ds_phr.shape
            args.crs = ds_phr.crs
            args.transform = ds_phr.transform
            args.rpc = ds_phr.tags(ns="RPC")
            

            ds_phr.close()
            del ds_phr
            
            # Store image in shared memmory
            key_phr = eoscale_manager.open_raster(raster_path = args.file_phr)
            
            # Get cloud mask if any
            if args.file_cloud_gml:
                cloud_mask_array = np.logical_not(
                    cloud_from_gml(args.file_cloud_gml, args.file_phr)   
                )
                #save cloud mask
                save_image(cloud_mask_array,
                    join(dirname(args.file_classif), "nocloud.tif"),
                    args.crs,
                    args.transform,
                    None,
                    args.rpc,
                    tags=args.__dict__,
                )
                mask_nocloud_key = eoscale_manager.open_raster(raster_path = join(dirname(args.file_classif), "nocloud.tif"))   
                
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
                                                           generate_output_profiles = single_bool_profile,
                                                           stable_margin= 0,
                                                           context_manager = eoscale_manager,
                                                           multiproc_context= "fork",
                                                           filter_desc= "Valid stack processing...")       
            
            
            
            ### Compute NDVI 
            if args.file_ndvi == None:
                key_ndvi = eoexe.n_images_to_m_images_filter(inputs = [key_phr, valid_stack_key[0]],
                                                               image_filter = compute_ndvi,
                                                               filter_parameters=args,
                                                               generate_output_profiles = single_int16_profile,
                                                               stable_margin= 0,
                                                               context_manager = eoscale_manager,
                                                               multiproc_context= "fork",
                                                               filter_desc= "NDVI processing...")
                if (args.save_mode != "none" and args.save_mode != "aux"):
                    eoscale_manager.write(key = key_ndvi[0], img_path = args.file_classif.replace(".tif","_NDVI.tif"))
            else:
                key_ndvi = [ eoscale_manager.open_raster(raster_path =args.file_ndvi) ]
                
            
            ### Compute NDWI        
            if args.file_ndwi == None:
                key_ndwi = eoexe.n_images_to_m_images_filter(inputs = [key_phr, valid_stack_key[0]],
                                                               image_filter = compute_ndwi,
                                                               filter_parameters=args,
                                                               generate_output_profiles = single_int16_profile,
                                                               stable_margin= 0,
                                                               context_manager = eoscale_manager,
                                                               multiproc_context= "fork",
                                                               filter_desc= "NDWI processing...")         
                if (args.save_mode != "none" and args.save_mode != "aux"):
                    eoscale_manager.write(key = key_ndwi[0], img_path = args.file_classif.replace(".tif","_NDWI.tif"))
            else:
                key_ndwi= [ eoscale_manager.open_raster(raster_path =args.file_ndwi) ]
  
            

            if args.file_watermask:
                key_watermask= eoscale_manager.open_raster(raster_path =args.file_watermask)
            else:
                profile = eoscale_manager.get_profile(key_phr)
                profile["count"] = 1
                profile["dtype"] = np.uint8
                key_watermask = eoscale_manager.create_image(profile)
                eoscale_manager.get_array(key=key_watermask).fill(0)


            if args.file_vegetationmask:
                key_vegmask= eoscale_manager.open_raster(raster_path =args.file_vegetationmask)
            else:
                profile = eoscale_manager.get_profile(key_phr)
                profile["count"] = 1
                profile["dtype"] = np.uint8
                key_vegmask = eoscale_manager.create_image(profile)
                eoscale_manager.get_array(key=key_vegmask).fill(0)
                

           

            time_stack = time.time()
            
            ################ Build samples #################
            
            if args.urban_raster:
                gt_key= eoscale_manager.open_raster(raster_path =args.urban_raster)

            else:
                args.urban_raster = join(dirname(args.file_classif), "wsf.tif")
                im_gt = wsf_recovery(args.file_phr, args.urban_raster, True)  
                gt_key= eoscale_manager.open_raster(raster_path =args.urban_raster)
            
            #Recover useful features
            valid_stack= eoscale_manager.get_array(valid_stack_key[0])
            local_gt= eoscale_manager.get_array(gt_key)
            file_filters= [eoscale_manager.open_raster(raster_path =args.files_layers[i]) for i in range(len(args.files_layers))]
            
            #Calcul of valid pixels
            nb_valid_pixels = np.count_nonzero(valid_stack)
            args.nb_valid_built_pixels = np.count_nonzero(np.logical_and(local_gt, valid_stack))
            args.nb_valid_other_pixels = nb_valid_pixels - args.nb_valid_built_pixels                                      

            samples = eoexe.n_images_to_m_scalars(inputs=[valid_stack_key[0],gt_key, key_phr, key_ndvi[0], key_ndwi[0]] + file_filters,  
                                                    image_filter = build_samples,   
                                                    filter_parameters = args,
                                                    nb_output_scalars = args.nb_valid_built_pixels+args.nb_valid_other_pixels,
                                                    context_manager = eoscale_manager,
                                                    concatenate_filter = concatenate_samples, 
                                                    output_scalars= [],
                                                    multiproc_context= "fork",
                                                    filter_desc= "Samples building processing...")       # samples=[y_samples, x_samples]
            
            
            time_samples = time.time()
            
            ################ Train classifier from samples #########
            
            classifier = RandomForestClassifier(
            n_estimators=args.nb_estimators, max_depth=args.max_depth, class_weight="balanced",
            random_state=0, n_jobs=args.nb_jobs
            )
            print("RandomForest parameters:\n", classifier.get_params(), "\n")
            samples=np.concatenate(samples[:]) 
            x_samples= samples[:,1:]
            y_samples= samples[:,0]
            
            print(x_samples.shape)
            train_classifier(classifier, x_samples, y_samples)
            print("Dump classifier to model_rf.dump")
            joblib.dump(classifier, "model_rf.dump")
            print_feature_importance(classifier, names_stack)
            gc.collect()
            
            
            ######### Predict  ################
            
            key_predict = eoexe.n_images_to_m_images_filter(inputs = [valid_stack_key[0], key_phr, key_ndvi[0], key_ndwi[0]] + file_filters,   
                                                           image_filter = RF_prediction,
                                                           filter_parameters= {"classifier": classifier},
                                                           generate_output_profiles = three_uint8_profile,
                                                           stable_margin= 0,
                                                           context_manager = eoscale_manager,
                                                           multiproc_context= "fork",
                                                           filter_desc= "RF prediction processing...")             
            time_random_forest = time.time()
            
            ######### Post_processing  ################  
            
            key_post_process = eoexe.n_images_to_m_images_filter([key_predict[0],key_phr, key_watermask, key_vegmask,gt_key,valid_stack_key[0]],   
                                                           image_filter = post_process,
                                                           filter_parameters= args,
                                                           generate_output_profiles = post_process_profile,
                                                           stable_margin= 20,
                                                           context_manager = eoscale_manager,
                                                           multiproc_context= "fork",
                                                           filter_desc= "Post processing...")

            
            # Save final mask (prediction + post-processing)
            final_classif = eoscale_manager.get_array(key_post_process[0])[0]
            save_image(
                final_classif,
                args.file_classif,
                args.crs,
                args.transform,
                255,
                args.rpc,
                dtype=np.dtype(np.uint8),
                tags=args.__dict__,
            )
            if args.save_mode == "debug":
                # Save auxilliary results : raw prediction, markers
                save_image(
                    eoscale_manager.get_array(key_post_process[0])[1],
                    join(dirname(args.file_classif), basename(args.file_classif).replace(".tif","_markers.tif")),
                    args.crs,
                    args.transform,
                    255,
                    args.rpc,
                    dtype=np.dtype(np.uint8),
                    tags=args.__dict__,
                )
                final_predict = eoscale_manager.get_array(key_predict[0])
                save_image(
                    final_predict[0],
                    join(dirname(args.file_classif), basename(args.file_classif).replace(".tif","_raw_predict.tif")),
                    args.crs,
                    args.transform,
                    255,
                    args.rpc,
                    tags=args.__dict__,
                )
                save_image(
                    final_predict[2],
                    join(dirname(args.file_classif), basename(args.file_classif).replace(".tif","_proba_urban.tif")),
                    args.crs,
                    args.transform,
                    255,
                    args.rpc,
                    tags=args.__dict__,
                )
            end_time = time.time()
            
            print("**** Urban mask for "+str(args.file_phr)+" (saved as "+str(args.file_classif)+") ****")
            print("Total time (user)       :\t"+convert_time(end_time-t0))
            print("- Build_stack           :\t"+convert_time(time_stack-t0))
            print("- Build_samples         :\t"+convert_time(time_samples-time_stack))
            print("- Random forest (total) :\t"+convert_time(time_random_forest-time_samples))
            print("- Post-processing       :\t"+convert_time(end_time-time_random_forest))
            print("***")    
        
        
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
