#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Compute water mask of PHR image with help of Pekel and Hand images."""

import argparse
import gc
import time
import traceback
from os.path import dirname, join

import numpy as np
import rasterio as rio
from skimage.filters.rank import maximum
from skimage.measure import label, regionprops
from skimage.morphology import area_closing, binary_closing, remove_small_holes, square, disk
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pylab import *

from slurp.tools import io_utils, utils
from slurp.tools import eoscale_utils as eo_utils
import eoscale.manager as eom
import eoscale.eo_executors as eoexe

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ModuleNotFoundError:
    print("Intel(R) Extension/Optimization for scikit-learn not found.")


def compute_pekel_mask(input_buffer: list, input_profiles: list, params: dict) -> list:
    """
    Compute Pekel mask regarding entry arguments

    :param list input_buffer: Pekel image [pekel_image]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: Pekel masks
    """
    if params["hand_strict"]:
        if not params["no_pekel_filter"]:
            [mask_pekel, mask_pekelxx] = utils.compute_mask(
                input_buffer[0],
                [params["thresh_pekel"], params["strict_thresh"]]
            )
        else:
            [mask_pekel, mask_pekelxx] = utils.compute_mask(
                input_buffer[0],
                [params["thresh_pekel"], params["strict_thresh"]]
            )
        return [mask_pekel, mask_pekelxx]

    elif not params["no_pekel_filter"]:
        [mask_pekel, mask_pekel0] = utils.compute_mask(
            input_buffer[0],
            [params["thresh_pekel"], 0])
    else:
        mask_pekel = utils.compute_mask(input_buffer[0], params["thresh_pekel"])
        mask_pekel0 = np.zeros(input_buffer[0].shape)

    return [mask_pekel, mask_pekel0]


def compute_hand_mask(input_buffer: list, input_profiles: list, params: dict) -> bool:
    """
    Compute Hand mask with one or multiple threshold values.

    :param list input_buffer: Hand image [hand_image]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: Hand mask
    """
    mask_hand = input_buffer[0] > params["thresh_hand"]

    # Do not learn in water surface (useful if image contains big water surfaces)
    # Add some robustness if hand_strict is not used
    # if args.hand_strict:
    # np.logical_not(np.logical_or(mask_hand, inputBuffer[1]), out=mask_hand)
    # else:
    # np.logical_not(mask_hand, out=mask_hand)
    np.logical_not(mask_hand, out=mask_hand)

    return mask_hand


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


def post_process(input_buffer: list, input_profiles: list, params: dict) -> list:
    """
    Compute some filters on the prediction image.

    :param list input_buffer: [im_predict, mask_hand, mask_pekel0, valid_stack + file filters*x]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: predict mask and post-processed mask
    """
    buffer_shape = input_buffer[0].shape

    # Filter with Hand
    if params["hand_filter"]:
        if not params["hand_strict"]:
            input_buffer[0][np.logical_not(input_buffer[1])] = 0
        else:
            print("\nWARNING: hand_filter and hand_strict are incompatible.")

    # Filter for final classification
    if len(input_buffer) > 4 or not params["no_pekel_filter"]:
        mask = np.zeros(buffer_shape, dtype=bool)
        if not params["no_pekel_filter"]:  # filter with pekel0
            mask = np.zeros(buffer_shape, dtype=bool)
            mask = np.logical_or(mask, input_buffer[2][0])  # problème de mask_pekel0 if "not defined"
        for i in range(len(input_buffer) - 4):  # Other classification files
            filter_mask = compute_filter(input_buffer[i + 4], "FILTER " + str(i))[0]
            mask = np.logical_or(mask, filter_mask)

        im_classif = mask_filter(input_buffer[0], mask)
    else:
        im_classif = input_buffer[0]

    # Closing
    if params["binary_closing"]:
        im_classif[0, :, :] = binary_closing(im_classif[0, :, :].astype(bool), disk(params["binary_closing"])).astype(
            np.uint8)
    elif params["area_closing"]:
        im_classif[0, :, :] = area_closing(im_classif[0, :, :], params["area_closing"], connectivity=2)
    elif params["remove_small_holes"]:
        im_classif[0, :, :] = remove_small_holes(
            im_classif[0, :, :].astype(bool), params["remove_small_holes"], connectivity=2
        ).astype(np.uint8)

    # Add nodata in im_classif
    im_classif[np.logical_not(input_buffer[3])] = 255
    im_classif[im_classif == 1] = params["value_classif"]

    im_predict = input_buffer[0]
    im_predict[np.logical_not(input_buffer[3])] = 255
    im_predict[im_predict == 1] = params["value_classif"]

    return [im_predict, im_classif]


def get_random_indexes_from_masks(nb_indexes, mask_1, mask_2):
    """Get random valid indexes from masks.
    Mask 1 is a validity mask
    """
    rows_idxs = []
    cols_idxs = []

    if nb_indexes != 0:
        nb_idxs = 0

        height = mask_1.shape[0]
        width = mask_1.shape[1]

        while nb_idxs < nb_indexes:
            np.random.seed(712)  # reproductible results
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
        indices = np.arange(0, len(rows) - 1, len(rows) / nb_samples).astype(np.uint16)

        s_rows = rows[indices]
        s_cols = cols[indices]
    else:
        s_rows = []
        s_cols = []

    return s_rows, s_cols


def get_smart_indexes_from_mask(nb_indexes, pct_area, minimum, mask):
    rows_idxs = []
    cols_idxs = []

    if nb_indexes != 0:
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
                np.random.seed(712)  # reproductible results
                row = np.random.randint(min_row, max_row)
                col = np.random.randint(min_col, max_col)

                if mask[row, col]:
                    rows_idxs.append(row)
                    cols_idxs.append(col)
                    nb_idxs += 1

    return rows_idxs, cols_idxs


def save_indexes(filename, water_idxs, other_idxs, shape, crs, transform, rpc, colormap):
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

    im_label_thresh = np.copy(im_label)
    im_label_thresh[np.logical_not(mask_ref)] = 0
    valid_labels = np.delete(np.unique(im_label_thresh), 0)

    im_filtered = np.zeros(np.shape(mask_ref), dtype=np.uint8)
    im_filtered[np.isin(im_label, valid_labels)] = 1

    return im_filtered


def print_feature_importance(classifier):
    """Compute feature importance."""
    feature_names = ["R", "G", "B", "NIR", "NDVI", "NDWI"]

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


def build_samples(input_buffer: list, input_profiles: list, params: dict) -> np.ndarray:
    """
    Build samples

    :param list input_buffer: [mask_pekel, valid_stack, mask_hand, im_phr, im_ndvi, im_ndwi]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: Retrieve number of pixels for each class
    """
    nb_valid_subset = np.count_nonzero(input_buffer[1])
    valid_water_pixels = np.logical_and(input_buffer[0], input_buffer[5] > params["ndwi_threshold"])

    nb_water_subset = np.count_nonzero(np.logical_and(valid_water_pixels, input_buffer[1]))
    nb_other_subset = nb_valid_subset - nb_water_subset

    # Ratio of pixel class compare to the full image ratio
    water_ratio = nb_water_subset / params["nb_valid_water_pixels"]
    other_ratio = nb_other_subset / params["nb_valid_other_pixels"]
    # Retrieve number of samples to create for each class in this subset
    nb_water_subsamples = round(water_ratio * params["nb_samples_water"])
    nb_other_subsamples = round(other_ratio * params["nb_samples_other"])

    # Prepare random water and other samples
    if params["nb_samples_auto"]:
        nb_water_subsamples = int(nb_water_subset * params["auto_pct"])
        nb_other_subsamples = int(nb_other_subset * params["auto_pct"])

    # Pekel samples
    if params["samples_method"] == "random":
        rows_pekel, cols_pekel = get_random_indexes_from_masks(
            nb_water_subsamples, input_buffer[1][0], input_buffer[0][0]
        )
        # Hand samples, always random (currently)
        rows_hand, cols_hand = get_random_indexes_from_masks(
            nb_other_subsamples, input_buffer[1][0], input_buffer[2][0]
        )

    elif params["samples_method"] == "smart":
        rows_pekel, cols_pekel = get_smart_indexes_from_mask(
            nb_water_subsamples,
            params["smart_area_pct"],
            params["smart_minimum"],
            np.logical_and(input_buffer[0][0], input_buffer[1][0]),
        )
        # Hand samples, always random (currently)
        rows_hand, cols_hand = get_random_indexes_from_masks(
            nb_other_subsamples, input_buffer[1][0], input_buffer[2][0]
        )

    elif params["samples_method"] == "grid":
        rows_pekel, cols_pekel = get_grid_indexes_from_mask(nb_water_subsamples, input_buffer[1], valid_water_pixels[0])

        # Hand samples, always random (currently)
        rows_hand, cols_hand = get_grid_indexes_from_mask(nb_other_subsamples, input_buffer[1], input_buffer[2])

    else:
        raise Exception("Sample method not accepted : use 'random', 'smart' or 'grid'")

    # All samples
    rows = np.concatenate((rows_pekel, rows_hand))
    cols = np.concatenate((cols_pekel, cols_hand))
    if params["save_mode"] == "debug":
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
            params["shape"],
            params["crs"],
            params["transform"],
            params["rpc"],
            colormap
        )

    # Prepare samples for learning
    im_stack = np.concatenate((input_buffer[3], input_buffer[4], input_buffer[5], input_buffer[0]), axis=0)
    samples = np.transpose(im_stack[:, rows.astype(np.uint16), cols.astype(np.uint16)])

    return samples  # [x_samples, y_samples]


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


def RF_prediction(input_buffer: list, input_profiles: list, params: dict) -> np.ndarray:
    """
    Random Forest prediction

    :param list input_buffer: [vhr_image, ndvi, ndwi, valid_stack]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments
    :returns: predicted mask
    """
    im_stack = np.concatenate((input_buffer[0], input_buffer[1], input_buffer[2]), axis=0)
    valid_mask = input_buffer[3].astype(bool)
    buffer_to_predict = np.transpose(im_stack[:, valid_mask[0]])

    classifier = params["classifier"]
    prediction = np.zeros(valid_mask[0].shape, dtype=np.uint8)
    if buffer_to_predict.shape[0] == 0:
        print(f"WARNING > zone with NO DATA")
    else:
        prediction[valid_mask[0]] = classifier.predict(buffer_to_predict)

    return prediction


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

    group1.add_argument("-pekel", action="store", dest="extracted_pekel", help="Pekel filename")
    group1.add_argument("-hand", action="store", dest="extracted_hand", help="Hand filename")
    group1.add_argument("-ndvi", action="store", dest="file_ndvi", help="NDVI filename")
    group1.add_argument("-ndwi", action="store", dest="file_ndwi", help="NDWI filename")
    group1.add_argument("-layers", nargs="+", action="store", dest="files_layers", metavar="FILE_LAYER",
                        help="Add layers as features used by learning algorithm")
    group1.add_argument("-filters", nargs="+", action="store", dest="file_filters",
                        help="Add files used in filtering (postprocessing)")
    group1.add_argument("-valid", action="store", dest="valid_stack", help="Validity mask")

    # Options
    group2.add_argument("-thresh_pekel", type=float, action="store", help="Pekel Threshold float (default is 50)")
    group2.add_argument("-hand_strict", action="store_true", help="Use not(pekelxx) for other (no water) samples")
    group2.add_argument("-thresh_hand", type=int, action="store", help="Hand Threshold int >= 0 (default is 25)")
    group2.add_argument("-strict_thresh", type=float, action="store", help="Pekel Threshold float (default is 50)",)
    group2.add_argument("-save_mode", choices=["none", "debug"], action="store",
                        help="Save all files (debug) or only output mask (none)")
    group2.add_argument("-simple_ndwi_threshold", action="store",
                        help="Compute water mask as a simple NDWI threshold - useful in arid places where no water is known by Peckel")
    group2.add_argument("-ndwi_threshold", type=float, action="store",
                        help="Threshold used when Pekel is empty in the area")

    # Samples
    group3.add_argument("-samples_method", choices=["smart", "grid", "random"], action="store",
                        help="Select method for choosing learning samples")

    group3.add_argument("-nb_samples_water", type=int, action="store",
                        help="Number of samples in water for learning (default is 2000)")

    group3.add_argument("-nb_samples_other", type=int, action="store",
                        help="Number of samples in other for learning (default is 10000)")

    group3.add_argument("-nb_samples_auto", action="store",
                        help="Auto select number of samples for water and other")

    group3.add_argument("-auto_pct", type=float, action="store",
                        help="Percentage of samples points, to use with -nb_samples_auto")

    group3.add_argument("-smart_area_pct", type=int, action="store",
                        help="For smart method, importance of area for selecting number of samples in each water surface.")

    group3.add_argument("-smart_minimum", type=int, action="store",
                        help="For smart method, minimum number of samples in each water surface.")

    group3.add_argument("-grid_spacing", type=int, action="store",
                        help="For grid method, select samples on a regular grid (40 pixels seems to be a good value)")

    group3.add_argument("-max_depth", type=int, action="store", help="Max depth of trees")

    group3.add_argument("-nb_estimators", type=int, action="store", help="Nb of trees in Random Forest")

    group3.add_argument("-n_jobs", type=int, action="store",
                        help="Nb of parallel jobs for Random Forest (1 is recommanded : use n_workers to optimize parallel computing)")

    # Post-processing
    group4.add_argument("-no_pekel_filter", action="store", 
                        help="Deactivate postprocess with pekel which only keeps surfaces already known by pekel")

    group4.add_argument("-hand_filter", action="store", 
                        help="Postprocess with Hand (set to 0 when hand > thresh), incompatible with hand_strict",)

    group4.add_argument("-binary_closing", type=int, action="store", help="Size of square structuring element",)
    group4.add_argument("-area_closing", type=int, action="store", help="Area closing removes all dark structures",)
    group4.add_argument("-remove_small_holes", type=int, action="store",
                        help="The maximum area, in pixels, of a contiguous hole that will be filled",)

    # Output
    group5.add_argument("-watermask", help="Output classification filename")
    group5.add_argument("-value_classif", type=int, action="store", help="Output classification value (default is 1)")

    # Parallel computing
    group6.add_argument("-max_mem", type=int, action="store", dest="max_memory",
                        help="Max memory permitted for the prediction of the Random Forest (in Gb)")

    group6.add_argument("-n_workers", type=int, action="store", help="Nb of CPU")

    return parser.parse_args()


################ Main function ################


def main():
    argparse_dict = vars(getarguments())

    # Read the JSON files
    keys = ['input', 'aux_layers', 'masks', 'ressources', 'water']
    argsdict = io_utils.read_json(argparse_dict["main_config"], keys, argparse_dict.get("user_config"))

    # Overload with manually passed arguments if not None
    for key in argparse_dict.keys():
        if argparse_dict[key] is not None:
            argsdict[key] = argparse_dict[key]

    print("JSON data loaded:")
    print(argsdict)
    args = argparse.Namespace(**argsdict)

    with eom.EOContextManager(nb_workers=args.n_workers, tile_mode=True) as eoscale_manager:

        try:

            t0 = time.time()

            ################ Build stack with all layers #######

            # Image PHR (numpy array, 4 bands, band number is first dimension),
            key_phr = eoscale_manager.open_raster(raster_path=args.file_vhr)
            profile_phr = eoscale_manager.get_profile(key_phr)
            eo_utils.print_dataset_infos(args.file_vhr, profile_phr, "PHR")

            args.nodata_phr = profile_phr["nodata"]
            args.shape = (profile_phr["height"], profile_phr["width"])
            args.crs = profile_phr["crs"]
            args.transform = profile_phr["transform"]
            args.rpc = None

            # Valid stack
            key_valid_stack = eoscale_manager.open_raster(raster_path=args.valid_stack)

            # NDXI
            key_ndvi = eoscale_manager.open_raster(raster_path=args.file_ndvi)
            key_ndwi = eoscale_manager.open_raster(raster_path=args.file_ndwi)

            time_stack = time.time()

            ################ Build samples ##################

            # Pekel
            key_pekel = eoscale_manager.open_raster(raster_path=args.extracted_pekel)
            pekel_profile = eoscale_manager.get_profile(key_pekel)
            eo_utils.print_dataset_infos(args.extracted_pekel, pekel_profile, "PEKEL")
            args.pekel_nodata = pekel_profile["nodata"]

            # Pekel valid masks
            mask_pekel = eoexe.n_images_to_m_images_filter(inputs=[key_pekel],
                                                           image_filter=compute_pekel_mask,
                                                           filter_parameters=vars(args),
                                                           generate_output_profiles=eo_utils.double_int_profile,
                                                           stable_margin=0,
                                                           context_manager=eoscale_manager,
                                                           multiproc_context="fork",
                                                           filter_desc="Pekel valid mask processing...")

            # If user wants a simple threshold on NDWI values, we don't select samples and launch learning/prediction step
            # If there are not enough water samples, we return a void mask
            not_enough_water_samples = False

            ### Check pekel mask
            # - if there are too few values : we threshold NDWI to detect water areas
            # - if there are even no "supposed water areas" : stop machine learning process (flag select_samples=False)
            local_mask_pekel = eoscale_manager.get_array(mask_pekel[0])
            if np.count_nonzero(local_mask_pekel) < args.nb_samples_water:
                # In case they are too few Pekel pixels, we prefer to threshold NDWI and skip samples selection
                # Alternative would be to select samples in a thresholded NDWI...
                not_enough_water_samples = True
                print("** WARNING ** not enough water samples are found in Pekel : return a void mask")

            # HAND
            key_hand = eoscale_manager.open_raster(raster_path=args.extracted_hand)
            hand_profile = eoscale_manager.get_profile(key_hand)
            eo_utils.print_dataset_infos(args.extracted_hand, hand_profile, "HAND")
            args.hand_nodata = hand_profile["nodata"]

            # Create HAND mask
            mask_hand = eoexe.n_images_to_m_images_filter(inputs=[key_hand],
                                                          image_filter=compute_hand_mask,
                                                          # args.hand_strict impossible because of mask_pekel0 not sure
                                                          filter_parameters=vars(args),
                                                          generate_output_profiles=eo_utils.single_float_profile,
                                                          stable_margin=0,
                                                          context_manager=eoscale_manager,
                                                          multiproc_context="fork",
                                                          filter_desc="Hand valid mask processing...")

            # Flag to command post-process
            do_post_process = True

            if args.simple_ndwi_threshold:
                # Simple NDWI threshold, but taking account valid stack to take care of NO_DATA values
                print("Simple threshold mask NDWI > " + str(args.ndwi_threshold))
                key_predict = eoexe.n_images_to_m_images_filter(inputs=[key_ndwi, key_valid_stack],
                                                                image_filter=utils.compute_mask_threshold,
                                                                filter_parameters={
                                                                    "threshold": 1000 * args.ndwi_threshold},
                                                                context_manager=eoscale_manager,
                                                                generate_output_profiles=eo_utils.single_uint8_profile,
                                                                multiproc_context="fork",
                                                                filter_desc="Simple NDWI threshold")

                time_random_forest = time.time()
                time_samples = time_random_forest
                do_post_process = False

            elif not_enough_water_samples:
                # We compute a void mask (0 everywhere, except for NO DATA values)
                # Tips : we threshold NDWI > 1000 : no pixel should be detected.
                key_predict = eoexe.n_images_to_m_images_filter(inputs=[key_ndwi, key_valid_stack],
                                                                image_filter=utils.compute_mask_threshold,
                                                                filter_parameters={"threshold": 1000},
                                                                context_manager=eoscale_manager,
                                                                generate_output_profiles=eo_utils.single_uint8_profile,
                                                                multiproc_context="fork",
                                                                filter_desc="Void mask")

                do_post_process = False

            else:
                # Nominal case : select samples, train, predict
                #
                # Sample selection
                valid_stack = eoscale_manager.get_array(key_valid_stack)
                nb_valid_pixels = np.count_nonzero(valid_stack)
                args.nb_valid_water_pixels = np.count_nonzero(np.logical_and(local_mask_pekel, valid_stack))
                args.nb_valid_other_pixels = nb_valid_pixels - args.nb_valid_water_pixels
                input_for_samples = [mask_pekel[0], key_valid_stack, mask_hand[0], key_phr, key_ndvi, key_ndwi]

                samples = eoexe.n_images_to_m_scalars(inputs=input_for_samples,
                                                      image_filter=build_samples,
                                                      filter_parameters=vars(args),
                                                      nb_output_scalars=args.nb_samples_water + args.nb_samples_other,
                                                      context_manager=eoscale_manager,
                                                      concatenate_filter=utils.concatenate_samples,
                                                      output_scalars=[],
                                                      multiproc_context="fork",
                                                      filter_desc="Samples building processing...")
                # samples=[x_samples, y_samples]

                time_samples = time.time()

                ################ Train classifier from samples ########
                classifier = RandomForestClassifier(
                    n_estimators=args.nb_estimators, max_depth=args.max_depth, random_state=712, n_jobs=1
                )
                print("RandomForest parameters:\n", classifier.get_params(), "\n")
                samples = np.concatenate(samples[:])  # A revoir si possible
                x_samples = samples[:, :-1]
                y_samples = samples[:, -1]
                train_classifier(classifier, x_samples, y_samples)
                print_feature_importance(classifier)
                gc.collect()

                ######### Predict  ################
                input_for_prediction = [key_phr, key_ndvi, key_ndwi, key_valid_stack]
                key_predict = eoexe.n_images_to_m_images_filter(inputs=input_for_prediction,
                                                                image_filter=RF_prediction,
                                                                filter_parameters={"classifier": classifier},
                                                                generate_output_profiles=eo_utils.single_float_profile,
                                                                stable_margin=0,
                                                                context_manager=eoscale_manager,
                                                                multiproc_context="fork",
                                                                filter_desc="RF prediction processing...")
                time_random_forest = time.time()

            if do_post_process:
                ######### Post_processing  ################
                file_filters = [
                    eoscale_manager.open_raster(raster_path=args.file_filters[i])
                    for i in range(len(args.file_filters))
                ]

                inputs_for_classif = [key_predict[0], mask_hand[0], mask_pekel[1], key_valid_stack] + file_filters
                im_classif = eoexe.n_images_to_m_images_filter(inputs=inputs_for_classif,
                                                               image_filter=post_process,
                                                               filter_parameters=vars(args),
                                                               generate_output_profiles=eo_utils.double_int_profile,
                                                               stable_margin=3,
                                                               context_manager=eoscale_manager,
                                                               multiproc_context="fork",
                                                               filter_desc="Post processing...")

                # Save predict and classif image
                eoscale_manager.write(key=im_classif[0], img_path=join(dirname(args.watermask), "predict.tif"))
                eoscale_manager.write(key=im_classif[1], img_path=args.watermask)  # classif
            else:
                # no post-process : we save the same mask with two different names for compatibility purpose
                eoscale_manager.write(key=key_predict[0], img_path=join(dirname(args.watermask), "predict.tif"))
                eoscale_manager.write(key=key_predict[0], img_path=args.watermask)  # classif

            end_time = time.time()

            print(f"**** Water mask for {args.file_vhr} (saved as {args.watermask}) ****")
            print("Total time (user)       :\t" + utils.convert_time(end_time - t0))
            print("- Build_stack           :\t" + utils.convert_time(time_stack - t0))
            if not args.simple_ndwi_threshold and not not_enough_water_samples:
                print("- Build_samples         :\t" + utils.convert_time(time_samples - time_stack))
                print("- Random forest (total) :\t" + utils.convert_time(time_random_forest - time_samples))
                print("- Post-processing       :\t" + utils.convert_time(end_time - time_random_forest))
            print("***")
            print("Max workers used for parallel tasks " + str(args.n_workers))

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
