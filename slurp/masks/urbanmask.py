#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Compute building and road masks from VHR images thanks to OSM layers """

import argparse
import json
import gc
import numpy as np
import otbApplication as otb
import random
import rasterio as rio
import time
import traceback

from os.path import dirname, join, basename
from skimage import segmentation
from skimage.filters import sobel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skimage.morphology import (binary_closing, binary_opening, binary_dilation, binary_erosion, remove_small_holes,
                                remove_small_objects, square, disk)

from slurp.prepare import aux_files as aux
from slurp.prepare.primitives import compute_ndvi, compute_ndwi
from slurp.tools import io_utils, utils, eoscale_utils as eo_utils
import eoscale.manager as eom
import eoscale.eo_executors as eoexe


try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ModuleNotFoundError:
    print("Intel(R) Extension/Optimization for scikit-learn not found.")


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

    
def get_grid_indexes_from_mask(nb_samples, valid_mask, mask_ground_truth):
    valid_samples = np.logical_and(mask_ground_truth, valid_mask).astype(np.uint8)
    _, rows, cols = np.where(valid_samples)

    if len(rows) >= nb_samples and nb_samples >= 1:
        # np.arange(0, len(rows) -1, ...) : to be sure to exclude index len(rows)
        # because in some cases (ex : 19871, 104 samples), last index is the len(rows)
        indices = np.arange(0, len(rows)-1, int(len(rows)/nb_samples))
        s_rows = rows[indices]
        s_cols = cols[indices]
    else:
        s_rows = []
        s_cols = []
        
    return s_rows, s_cols


def build_samples(input_buffer: list, input_profiles: list, params: dict) -> np.ndarray:
    """
    Build samples

    :param list input_buffer: [valid_stack, gt, im_phr, im_ndvi, im_ndwi] + files_layers
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: Retrieve number of pixels for each class
    """
    # Beware that WSF ground truth contains 0 (non building), 255 (building) but sometimes 1 (invalid pixels ?)
    mask_building_before_erosion = np.where(input_buffer[1] == params["value_classif"], True, False)
    mask_building = [binary_erosion(mask_building_before_erosion[0], disk(params["binary_closing"]))]
    mask_non_building = np.where(input_buffer[1] == 0, True, False)
    
    # Retrieve number of pixels for each class
    nb_valid_subset = np.count_nonzero(input_buffer[0])
    nb_built_subset = np.count_nonzero(np.logical_and(input_buffer[1], input_buffer[0]))
    nb_other_subset = nb_valid_subset - nb_built_subset
    # Ratio of pixel class compare to the full image ratio
    urban_ratio = nb_built_subset / params["nb_valid_built_pixels"]
    other_ratio = nb_other_subset / params["nb_valid_other_pixels"]
    # Retrieve number of samples to create for each class in this subset 
    nb_urban_subsamples = round(urban_ratio * params["nb_samples_urban"])
    nb_other_subsamples = round(other_ratio * params["nb_samples_other"])

    if nb_urban_subsamples > 0:
        # Building samples
        rows, cols = get_grid_indexes_from_mask(nb_urban_subsamples, input_buffer[0][0], mask_building)

        if nb_other_subsamples > 0:
            rows_nob, cols_nob = get_grid_indexes_from_mask(nb_other_subsamples, input_buffer[0][0], mask_non_building)
            rows = np.concatenate((rows, rows_nob), axis=0)
            cols = np.concatenate((cols, cols_nob), axis=0)
    else:
        if nb_other_subsamples > 0:
            rows, cols = get_grid_indexes_from_mask(nb_other_subsamples, input_buffer[0][0], mask_non_building)
        else:
            rows = []
            cols = []

    # Prepare samples for learning
    im_stack = np.concatenate((input_buffer[1:]), axis=0)  # TODO : gérer les files_layers optionnels
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


def RF_prediction(input_buffer: list, input_profiles: list, params: dict) -> np.ndarray:
    """
    Random Forest prediction

    :param list input_buffer: [valid_stack, vhr_image, ndvi, ndwi, valid_stack] + file_layers
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: predicted mask
    """
    im_stack = np.concatenate((input_buffer[1:]), axis=0)
    buffer_to_predict = np.transpose(im_stack[:, input_buffer[0][0]])
    # buffer_to_predict are non NODATA pixels, defined by all the primitives (R-G-B-NIR-NDVI-NDWI-[+ features]

    classifier = params["classifier"]
    if buffer_to_predict.shape[0] > 0:
        proba = classifier.predict_proba(buffer_to_predict)
        # Prediction, inspired by sklearn code to predict class
        res_classif = classifier.classes_.take(np.argmax(proba, axis=1), axis=0)
        res_classif[res_classif == 255] = 1

        prediction = np.zeros((3, input_buffer[0].shape[1], input_buffer[0].shape[2]))
        # Class predicted
        prediction[0][input_buffer[0][0]] = res_classif
        # Proba for class 0 (background)
        prediction[1][input_buffer[0][0]] = 100 * proba[:, 0]
        # Proba for class 1 (buildings)
        prediction[2][input_buffer[0][0]] = 100 * proba[:, 1]

    else:
        ### corner case : only NO_DATA !
        prediction = np.zeros((3, input_buffer[0].shape[1], input_buffer[0].shape[2]))
        prediction[0][input_buffer[0][0]].fill(255)
        prediction[1][input_buffer[0][0]].fill(0)
        prediction[2][input_buffer[0][0]].fill(0)

    return prediction


def post_process(input_buffer: list, input_profiles: list, params: dict) -> np.ndarray:
    """
    Compute some filters on the prediction image.

    :param list input_buffer: [im_predict, im_phr, watermask, vegetationmask, shadowmask, gt, valid_stack]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: post-processed mask
    """
    # Clean
    # im_classif = clean(params, inputBuffer[0][0])

    # Watershed regulation
    final_mask, markers, edges = watershed_regul(params, input_buffer[0][0], input_buffer)

    # Add nodata in final_mask (input_buffer[6] : valid mask)
    final_mask[np.logical_not(input_buffer[6][0])] = 255

    res_int = np.zeros((3, input_buffer[1].shape[1], input_buffer[1].shape[2]))
    res_int[0] = final_mask
    res_int[1] = markers
    res_int[2] = input_buffer[0][0]  # im_classif

    return res_int


def watershed_regul(params: dict, clean_predict: np.ndarray, input_buffer: list):
    """
    Compute watershed regulation.

    :param dict params: dictionary of arguments
    :param np.ndarray clean_predict: input image
    :param list input_buffer: [im_predict, im_phr, watermask, vegetationmask, shadowmask, gt, valid_stack]
    :returns: post-processed image
    """
    # Compute mono image from RGB image
    im_mono = 0.29 * input_buffer[1][0] + 0.58 * input_buffer[1][1] + 0.114 * input_buffer[1][2]

    # compute gradient
    edges = sobel(im_mono)

    del im_mono

    # markers map : -1, 1 and 2 : probable background, buildings or false positive
    # input_buffer[0] = proba of building class
    markers = np.zeros_like(input_buffer[0][0])

    """
    weak_detection = np.logical_and(input_buffer[0][2] > 50, input_buffer[0][2] < args.confidence_threshold)
    true_negative = np.logical_and(binary_closing(input_buffer[5][0], disk(10)) == 255, weak_detection)
    markers[weak_detection] = 3
    """

    #  probable_buildings = np.logical_and(input_buffer[0][2] > args.confidence_threshold, clean_predict == 1)
    probable_background = np.logical_and(input_buffer[0][2] < 40, clean_predict == 0)
    ground_truth_eroded = binary_erosion(input_buffer[5][0] == 255, disk(5))
    probable_buildings = np.logical_and(ground_truth_eroded, input_buffer[0][2] > 50)

    """
    ground_truth_eroded = binary_erosion(input_buffer[5][0]==255, disk(5))
    # If WSF = 1
    probable_buildings = np.logical_and(input_buffer[0][2] > 70, ground_truth_eroded)
    #probable_background = np.logical_and(input_buffer[0][2] < 40, ground_truth_eroded)

    no_WSF = binary_dilation(input_buffer[5][0]==0, disk(5)) 
    false_positive = np.logical_and(no_WSF, input_buffer[0][2] > params["confidence_threshold"])

    # note : all other pixels are 0
    markers[false_positive] = 2
    """

    confident_buildings = np.logical_and(ground_truth_eroded, input_buffer[0][2] > params["confidence_threshold"])

    markers[probable_background] = 4
    markers[probable_buildings] = 1
    markers[confident_buildings] = 2

    if params["file_shadowmask"]:
        # shadows (note : 2 are "cleaned / big shadows", 1 is raw shadow detection)
        markers[binary_erosion(input_buffer[4][0] == 2, disk(5))] = 8

    '''
    if params["file_vegetationmask"]:
        # vegetation
        markers[input_buffer[3][0] > params["vegmask_max_value"]] = 7
    if params["file_watermask"]:
        # water
        markers[input_buffer[2][0] == 1] = 6
    '''

    if params["remove_false_positive"]:
        ground_truth = input_buffer[5][0]
        # mark as false positive pixels with high confidence but not covered by dilated ground truth
        # TODO : check if we can reduce radius for dilation
        false_positive = np.logical_and(binary_dilation(ground_truth, disk(10)) == 0,
                                        input_buffer[0][2] > params["confidence_threshold"])
        markers[false_positive] = 3
        del ground_truth, false_positive

    # watershed segmentation
    # seg[np.where(seg>3, True, False)] = 0
    # markers[np.where(markers > 3)] = 0
    seg = segmentation.watershed(edges, markers)

    seg[np.where(seg > 3, True, False)] = 0
    seg[np.where(seg == 2, True, False)] = 1

    # TODO : check if we can remove/reduce this opening

    seg[binary_closing(seg == 1, disk(params["binary_closing"]))] = 1
    seg[binary_opening(seg == 1, disk(params["binary_closing"]))] = 1

    # markers[binary_opening(input_buffer[4][0] == 2, disk(10))] = 8

    if params["remove_small_holes"]:
        res = remove_small_holes(
            seg.astype(bool), params["remove_small_holes"], connectivity=2
        ).astype(np.uint8)
        seg = np.multiply(res, seg)

    # remove small artefacts : TODO seg contains 1, 2, 3, 4 values...
    # params["remove_small_objects"] = False
    if params["remove_small_objects"]:
        res = remove_small_objects(seg.astype(bool), params["remove_small_objects"], connectivity=2).astype(np.uint8)
        # res is either 0 or 1 : we multiply by seg to keep 0/1/2 classes
        seg = np.multiply(res, seg)

    return seg, markers, edges


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
    
    parser.add_argument("main_config", help="First JSON file, load basis arguments")
    parser.add_argument("-user_config", help="Second JSON file, overload basis arguments if keys are the same")
    parser.add_argument("-file_vhr", help="PHR filename")

    parser.add_argument("-red", help="Red band index")
    parser.add_argument("-nir", help="NIR band index")
    parser.add_argument("-green",help="green band index")

    
    parser.add_argument(
        "-save",
        choices=["none", "prim", "aux", "all", "debug"],
        required=False,
        action="store",
        dest="save_mode",
        help="Save all files (debug), only primitives (prim), only wsf (aux), primitives and wsf (all) or only output mask (none)",
    )

    parser.add_argument(
        "-ndvi",
        required=False,
        action="store",
        dest="file_ndvi",
        help="NDVI filename (computed if missing option)",
    )

    parser.add_argument(
        "-ndwi",
        required=False,
        action="store",
        dest="file_ndwi",
        help="NDWI filename (computed if missing option)",
    )
    
    parser.add_argument(
        "-watermask",
        required=False,
        action="store",
        dest="watermask",
        help="Watermask filename : urban mask will be learned & predicted, excepted on water areas"
    )
    
    parser.add_argument(
        "-vegetationmask",
        required=False,
        action="store",
        dest="vegetationmask",
        help="Vegetation mask filename : urban mask will be learned & predicted, excepted on vegetated areas"
    )

    parser.add_argument(
        "-vegmask_max_value",
        required=False,
        type=int,
        action="store",
        dest="vegmask_max_value",
        help="Vegetation mask value for vegetated areas : all pixels with lower value will be predicted"
    )

    parser.add_argument(
        "-shadowmask",
        required=False,
        action="store",
        dest="shadowmask",
        help="Shadowmask filename : big shadow areas will be marked as background"
    )

    parser.add_argument(
        "-post_process",
        required=False,
        action="store_true",
        dest="post_process",
        help="Post-process urban mask : apply morphological operations and regularize building shapes (watershed regularization)"
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
        required=False,
        action="store",
        dest="files_layers",
        metavar="FILE_LAYER",
        help="Add layers as features used by learning algorithm",
    )
    
    parser.add_argument(
        "-urban_raster",
        required=False,
        action="store",
        dest="urban_raster",
        help="Ground Truth (could be OSM, WSF). By default, WSF is automatically retrieved"
    )
    
    parser.add_argument(
        "-nb_classes",
        type=int,
        required=False,
        action="store",
        dest="nb_classes",
        help="Nb of classes in the ground-truth (1 by default - buildings only. Can be fix to 2 to classify buildings/roads"
    )

    parser.add_argument(
        "-urbanmask",
        help="Output classification filename (default is classif.tif)",
    )

    parser.add_argument(
        "-value_classif",
        type=int,
        required=False,
        action="store",
        dest="value_classif",
        help="Input ground truth class to consider in the input ground truth (default is 255 for WSF)",
    )

    parser.add_argument(
        "-nb_samples_urban",
        type=int,
        required=False,
        action="store",
        dest="nb_samples_urban",
        help="Number of samples in buildings for learning (default is 1000)",
    )

    parser.add_argument(
        "-nb_samples_other",
        type=int,
        required=False,
        action="store",
        dest="nb_samples_other",
        help="Number of samples in other for learning (default is 5000)",
    )

    parser.add_argument(
        "-max_depth",
        type=int,
        required=False,
        action="store",
        dest="max_depth",
        help="Max depth of trees"
    )

    parser.add_argument(
        "-nb_estimators",
        type=int,
        required=False,
        action="store",
        dest="nb_estimators",
        help="Nb of trees in Random Forest"
    )

    parser.add_argument(
        "-n_jobs",
        type=int,
        required=False,
        action="store",
        dest="n_jobs",
        help="Nb of parallel jobs for Random Forest"
    )
    
    parser.add_argument(
        "-n_workers",
        type=int,
        required=False,
        action="store",
        dest="n_workers",
        help="Nb of CPU"
    )
    
    parser.add_argument(
        "-random_seed",
        type=int,
        required=False,
        action="store",
        dest="random_seed",
        help="Fix the random seed for samples selection",
    )

    parser.add_argument(
        "-binary_closing",
        type=int,
        required=False,
        action="store",
        dest="binary_closing",
        help="Size of disk structuring element (erode GT before picking-up samples)"
    ) 

    parser.add_argument(
        "-binary_opening",
        type=int,
        required=False,
        action="store",
        dest="binary_opening",
        help="Size of square structuring element"
    )

    parser.add_argument(
        "-binary_dilation",
        type=int,
        required=False,
        action="store",
        dest="binary_dilation",
        help="Size of disk structuring element (dilate non vegetated areas)"
    )

    parser.add_argument(
        "-remove_small_objects",
        type=int,
        required=False,
        action="store",
        dest="remove_small_objects",
        help="The minimum area, in pixels, of the objects to detect",
    )
    
    parser.add_argument(
        "-remove_small_holes",
        type=int,
        required=False,
        action="store",
        dest="remove_small_holes",
        help="The minimum area, in pixels, of the holes to fill",
    )
    
    parser.add_argument(
        "-remove_false_positive",
        required=False,
        action="store_true",
        dest="remove_false_positive",
        help="Will dilate and use input ground-truth as mask to filter false positive from initial prediction"
    )
    
    parser.add_argument(
        "-confidence_threshold",
        type=int,
        required=False,
        action="store",
        dest="confidence_threshold",
        help="Confidence threshold to consider true positive in regularization step (85 by default)"
    )
    
    return parser.parse_args()


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
            argsdict.update(full_args['urban'])

            # a effacer après migration du pre-processing:
            argsdict.update(full_args['pre_process'])

    except FileNotFoundError:
        print(f"File {arg_file_path_1} not found.")
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
                    if k in ['input','aux_layers','masks','ressources', 'urban']:
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
    
    
    with eom.EOContextManager(nb_workers=args.n_workers, tile_mode=True) as eoscale_manager:
       
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
            ds_phr = rio.open(args.file_vhr)
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
            key_phr = eoscale_manager.open_raster(raster_path=args.file_vhr)
            
            # Get cloud mask if any
            if args.file_cloud_gml:
                cloud_mask_array = np.logical_not(
                    aux.cloud_from_gml(args.file_cloud_gml, args.file_vhr)
                )
                # save cloud mask
                io_utils.save_image(
                    cloud_mask_array,
                    join(dirname(args.urbanmask), "nocloud.tif"),
                    args.crs,
                    args.transform,
                    None,
                    args.rpc,
                    tags=args.__dict__,
                )
                mask_nocloud_key = eoscale_manager.open_raster(raster_path=join(dirname(args.urbanmask), "nocloud.tif"))
                
            else:
                # Get profile from im_phr
                profile = eoscale_manager.get_profile(key_phr)
                profile["count"] = 1
                profile["dtype"] = np.uint8
                mask_nocloud_key = eoscale_manager.create_image(profile)
                eoscale_manager.get_array(key=mask_nocloud_key).fill(1)

            if args.watermask:
                key_watermask = eoscale_manager.open_raster(raster_path=args.watermask)
            else:
                profile = eoscale_manager.get_profile(key_phr)
                profile["count"] = 1
                profile["dtype"] = np.uint8
                key_watermask = eoscale_manager.create_image(profile)
                eoscale_manager.get_array(key=key_watermask).fill(0)

            if args.vegetationmask:
                key_vegmask = eoscale_manager.open_raster(raster_path=args.vegetationmask)
            else:
                profile = eoscale_manager.get_profile(key_phr)
                profile["count"] = 1
                profile["dtype"] = np.uint8
                key_vegmask = eoscale_manager.create_image(profile)
                eoscale_manager.get_array(key=key_vegmask).fill(0)
                
            if args.shadowmask:
                key_shadowmask = eoscale_manager.open_raster(raster_path=args.shadowmask)
            else:
                profile = eoscale_manager.get_profile(key_phr)
                profile["count"] = 1
                profile["dtype"] = np.uint8
                key_shadowmask = eoscale_manager.create_image(profile)
                eoscale_manager.get_array(key=key_shadowmask).fill(0)

            if args.extracted_wsf:
                gt_key = eoscale_manager.open_raster(raster_path=args.extracted_wsf)

            else:
                args.extracted_wsf = join(dirname(args.urbanmask), "wsf.tif")
                im_gt = aux.wsf_recovery(args.file_vhr, args.wsf, args.extracted_wsf, True)
                gt_key = eoscale_manager.open_raster(raster_path=args.extracted_wsf)
            
            # Global validity mask construction
            input_for_valid_stack = [key_phr, mask_nocloud_key, key_vegmask, key_watermask]
            valid_stack_key = eoexe.n_images_to_m_images_filter(inputs=input_for_valid_stack,
                                                                image_filter=utils.compute_valid_stack_masks,
                                                                filter_parameters=vars(args),
                                                                generate_output_profiles=eo_utils.single_old_bool_profile,
                                                                stable_margin=0,
                                                                context_manager=eoscale_manager,
                                                                multiproc_context="fork",
                                                                filter_desc="Valid stack processing...")
            
            ### Compute NDVI 
            if args.file_ndvi is None:
                key_ndvi = eoexe.n_images_to_m_images_filter(inputs=[key_phr, valid_stack_key[0]],
                                                             image_filter=compute_ndvi,
                                                             filter_parameters=vars(args),
                                                             generate_output_profiles=eo_utils.single_int16_profile,
                                                             stable_margin=0,
                                                             context_manager=eoscale_manager,
                                                             multiproc_context="fork",
                                                             filter_desc="NDVI processing...")
                if args.save_mode != "none" and args.save_mode != "aux":
                    eoscale_manager.write(key=key_ndvi[0], img_path=args.urbanmask.replace(".tif", "_NDVI.tif"))
            else:
                key_ndvi = [eoscale_manager.open_raster(raster_path=args.file_ndvi)]
            
            ### Compute NDWI        
            if args.file_ndwi is None:
                key_ndwi = eoexe.n_images_to_m_images_filter(inputs=[key_phr, valid_stack_key[0]],
                                                             image_filter=compute_ndwi,
                                                             filter_parameters=vars(args),
                                                             generate_output_profiles=eo_utils.single_int16_profile,
                                                             stable_margin=0,
                                                             context_manager=eoscale_manager,
                                                             multiproc_context="fork",
                                                             filter_desc="NDWI processing...")
                if args.save_mode != "none" and args.save_mode != "aux":
                    eoscale_manager.write(key=key_ndwi[0], img_path=args.urbanmask.replace(".tif", "_NDWI.tif"))
            else:
                key_ndwi = [eoscale_manager.open_raster(raster_path=args.file_ndwi)]
  
            time_stack = time.time()
            
            ################ Build samples #################
                                   
            #Recover useful features
            valid_stack = eoscale_manager.get_array(valid_stack_key[0])
            local_gt = eoscale_manager.get_array(gt_key)
            file_filters = [
                eoscale_manager.open_raster(raster_path=args.files_layers[i])
                for i in range(len(args.files_layers))
            ]
            
            # Calcul of valid pixels
            nb_valid_pixels = np.count_nonzero(valid_stack)
            args.nb_valid_built_pixels = np.count_nonzero(np.logical_and(local_gt, valid_stack))
            args.nb_valid_other_pixels = nb_valid_pixels - args.nb_valid_built_pixels                                      

            if args.nb_valid_built_pixels > 0 and args.nb_valid_other_pixels > 0:
                ##### Nominal case : Ground Truth contains some pixels marked as building.  #####
                input_for_samples = [valid_stack_key[0], gt_key, key_phr, key_ndvi[0], key_ndwi[0]] + file_filters
                samples = eoexe.n_images_to_m_scalars(inputs=input_for_samples,
                                                      image_filter=build_samples,
                                                      filter_parameters=vars(args),
                                                      nb_output_scalars=args.nb_valid_built_pixels+args.nb_valid_other_pixels,
                                                      context_manager=eoscale_manager,
                                                      concatenate_filter=utils.concatenate_samples,
                                                      output_scalars=[],
                                                      multiproc_context="fork",
                                                      filter_desc="Samples building processing...")
                # samples=[y_samples, x_samples]

                time_samples = time.time()

                ################ Train classifier from samples #########

                classifier = RandomForestClassifier(
                    n_estimators=args.nb_estimators, max_depth=args.max_depth, class_weight="balanced",
                    random_state=0, n_jobs=args.n_jobs
                )
                print("RandomForest parameters:\n", classifier.get_params(), "\n")
                samples = np.concatenate(samples[:])
                x_samples = samples[:, 1:]
                y_samples = samples[:, 0]

                train_classifier(classifier, x_samples, y_samples)
                print_feature_importance(classifier, names_stack)
                gc.collect()

                ######### Predict  ################
                input_for_prediction = [valid_stack_key[0], key_phr, key_ndvi[0], key_ndwi[0]] + file_filters
                key_predict = eoexe.n_images_to_m_images_filter(inputs=input_for_prediction,
                                                                image_filter=RF_prediction,
                                                                filter_parameters={"classifier": classifier},
                                                                generate_output_profiles=eo_utils.three_uint8_profile,
                                                                stable_margin=0,
                                                                context_manager=eoscale_manager,
                                                                multiproc_context="fork",
                                                                filter_desc="RF prediction processing...")
                time_random_forest = time.time()

                final_predict = eoscale_manager.get_array(key_predict[0])
                io_utils.save_image(
                        final_predict[2],
                        join(dirname(args.urbanmask), basename(args.urbanmask).replace(".tif", "_proba.tif")),
                        args.crs,
                        args.transform,
                        255,
                        args.rpc,
                        tags=args.__dict__,
                )
                if args.save_mode == "debug":
                    io_utils.save_image(
                        final_predict[0],
                        join(dirname(args.urbanmask), basename(args.urbanmask).replace(".tif", "_raw_predict.tif")),
                        args.crs,
                        args.transform,
                        255,
                        args.rpc,
                        tags=args.__dict__,
                    )

                ######### Post_processing  ################  
                if args.post_process is True:
                    inputs_for_post_process = [
                        key_predict[0],
                        key_phr,
                        key_watermask,
                        key_vegmask,
                        key_shadowmask,
                        gt_key,
                        valid_stack_key[0]
                    ]
                    key_post_process = eoexe.n_images_to_m_images_filter(inputs=inputs_for_post_process,
                                                                         image_filter=post_process,
                                                                         filter_parameters=vars(args),
                                                                         generate_output_profiles=eo_utils.three_uint8_profile,
                                                                         stable_margin=20,
                                                                         context_manager=eoscale_manager,
                                                                         multiproc_context="fork",
                                                                         filter_desc="Post processing...")
                
                    # Save final mask (prediction + post-processing)
                    eoscale_manager.write(key=key_post_process[0][0], img_path=args.urbanmask)
                    eoscale_manager.write(
                        key=key_post_process[0][2],
                        img_path=join(
                            dirname(args.urbanmask),
                            basename(args.urbanmask).replace(".tif", "_clean.tif")
                        )
                    )
                    
                    if args.save_mode == "debug":
                        # Save auxilliary results : raw prediction, markers
                        eoscale_manager.write(
                            key=key_post_process[0][1],
                            img_path=join(
                                dirname(args.urbanmask),
                                basename(args.urbanmask).replace(".tif", "_markers.tif")
                            )
                        )

                end_time = time.time()

                print(f"**** Urban mask for {args.file_vhr} (saved as {args.urbanmask}) ****")
                print("Total time (user)       :\t"+convert_time(end_time-t0))
                print("- Build_stack           :\t"+convert_time(time_stack-t0))
                print("- Build_samples         :\t"+convert_time(time_samples-time_stack))
                print("- Random forest (total) :\t"+convert_time(time_random_forest-time_samples))
                if args.post_process is True:
                    print("- Post-processing       :\t"+convert_time(end_time-time_random_forest))
                print("***")   
                
            elif args.nb_valid_built_pixels > 0:
                #### Corner case : no "non building pixels"
                print(f"**** Only urban areas in {args.file_vhr} -> mask saved as {args.urbanmask} ****")
                
                profile = eoscale_manager.get_profile(key_phr)
                profile["count"] = 1
                profile["dtype"] = np.uint8
                profile["nodata"] = 255
                final_classif_key = eoscale_manager.create_image(profile)
                eoscale_manager.get_array(key=final_classif_key).fill(100)
                
                # Save final mask (prediction + post-processing)
                eoscale_manager.write(
                    key=final_classif_key,
                    img_path=join(
                        dirname(args.urbanmask),
                        basename(args.urbanmask).replace(".tif", "_proba.tif")
                    )
                )
                
            else:
                #### Corner case : no "building pixels" --> void mask (0)
                print(f"**** No urban areas in {args.file_vhr} -> void mask saved as {args.urbanmask} ****")
                
                profile = eoscale_manager.get_profile(key_phr)
                profile["count"] = 1
                profile["dtype"] = np.uint8
                profile["nodata"] = 255
                final_classif_key = eoscale_manager.create_image(profile)
                eoscale_manager.get_array(key=final_classif_key).fill(0)
                
                # Save final mask (prediction + post-processing)
                eoscale_manager.write(
                    key=final_classif_key,
                    img_path=join(
                        dirname(args.urbanmask),
                        basename(args.urbanmask).replace(".tif", "_proba.tif")
                    )
                )

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
