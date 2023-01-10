#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Compute building and road masks from VHR images thanks to OSM layers """

# Base from Xavier RAVE co3d_mask.py for water detection
# Adaptations to extract ground truth from OSM (shapefile) layers or raster building mask
# YT: TODO : fixer la valeur des masques en entrée
# YT: TODO : revoir arguments
# YT: TODO : distinguer deux modes : masque(s) complémentaire(s) pour ajouter des échantillons non buildings
#            et mode "nettoyage" ou le masque complémentaire sert à nettoyer le résultat

# Code linted with python linter pylint, score > 9

# XR: DONE: Utilisation de la librairie optimisée Intel
# XR: DONE: Correction typage du fichier de sortie SuperImpose
# XR: DONE: Ajout writerpctags pour OTB superimpose pour images sans geotags mais avec PHRDIMAP.XML
# XR: DONE: Utilisation de l'interpolateur NN pour OTB superimpose
# XR: DONE: Sauvegarde des points d'apprentissage dans un fichier pour analyse
# XR: DONE: Ajout d'une option use_rgb_layers pour ajouter les bandes rgb dans la stack d'apprentissage
# XR: DONE: Ajout d'une option -esri pour post processing avec carte ESRI
# XR: DONE: Ajout prise en compte des RPC
# XR: DONE: Superimpose Ajouter -elev.dem et -elev.geoid
# XR: TODO: Utiliser méthodes de Boosting
# XR: TODO: Utiliser l'attributs rasterio ds.descriptions=("",...) (liste de string)
# XR: TODO: Utiliser rasterio .tags() et .update_tags(ns='namespace', ...)
# XR: TODO: Ajouter des métadonnées dans le masque de sortie (valeur thresh, rgb, ...)
# XR: TODO: traiter le cas où il y a très peu de point dans le masque pekel (ajuster le seuil)
# XR: TODO: Mettre un facteur d'échelle modifiable sur le calcul des ndxi
# XR: DONE: Afficher l'importance des features
# XR: DONE: Optim1, pour calcul des echantillons non eau, utiliser le masque "hand and not pekel0"
# XR: DONE: Ajouter affichage d'un estimateur du Random Forest
# XR: DONE: Faire le post processing sur la carte pekel non seuillée
# XR: DONE: Mettre le ndvi et ndwi sur la plage [-1000,1000]
# XR: DONE: Ajouter le filtrage Hand (mettre à zéro les pixels non inondables)
# XR: TODO: Ajouter une colormap dans le predict.tif et le classif.tif
# XR: DONE: Ajouter la possibilité de lire le ndvi et ndwi dans un fichier
# XR: DONE: Ajouter une option pour activer ou non le post processing
# XR: DONE: Ajouter la possibilité de choisir la valeur du masque de sortie
# XR: DONE: Corriger l'utilisation du mask HAND, on ne garde que les zones inondables
# XR: DONE: Implémenter l'ajout de layers dans la stack pour le Random Forest
# XR: TODO: Dans superimpose, utiliser le type de l'image en entrée
# XR: TODO: Utiliser rasterio pour faire le superimpose ?
# XR: DONE: Ajouter un masque de validité global
# XR: DONE: Ajouter un masque de validité pekel
# XR: DONE: Ajouter un masque de validité hand
# XR: DONE: Ajouter un masque de validité dans le fonction de calcul du ndvi, ndwi
# XR: DONE: Gestion des divisions par 0 dans le calcul du ndvi, ndwi
# XR: DONE: Gestion des nan de la calcul des ndvi et ndwi
# XR: DONE: Supprimer la boucle pour le calcul des echantillons
# XR: DONE: Utiliser des int16 pour calcul ndvi et ndwi
# XR: DONE: KO: Utiliser des int8 pour calcul ndvi et ndwi (pb QGIS)
# XR: DONE: Resoudre pb calcul ndvi avec np.uint16
# XR: DONE: Mettre le crs + transform dans les images sauvegardées
# XR: DONE: Implementer la fonction save_image, avec compression des données
# XR: DONE: Implementer la fonction show_images
# XR: DONE: Implementer la fonction de filtrage post_process
# XR: DONE: Fusionner les masques nodata de l'image PHR
# XR: DONE: Utiliser rasterio pour la generation de la sortie
# XR: DONE: Ajouter le ndwi dans la stack PHR
# XR: DONE: Ajouter les options à la fonction classify
# XR: DONE: Ameliorer/simplifier la fonction hand_recovery
# XR: DONE: Ameliorer/simplifier la fonction pekel_recovery

import argparse
import gc
import time
import traceback
from os.path import dirname, join
from subprocess import call

import joblib
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from skimage.measure import label
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
import random
from slum.tools import io_utils

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ModuleNotFoundError:
    print("Intel(R) Extension/Optimization for scikit-learn not found.")


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

    # axe.plot(np.arange(-1000, 1001, step=10), hist1, color="blue", label=title1, **kwargs)
    # axe.plot(np.arange(-1000, 1001, step=10), hist2, color="red", label=title2, **kwargs)
    # axe.plot(np.arange(-1000, 1001, step=10), hist3, color="yellow", label=title3, **kwargs)
    # axe.plot(np.arange(-1000, 1001, step=10), hist4, color="green", label=title4, **kwargs)

    axe.plot(np.arange(-1000, 1001, step=10), hist1, label=title1, **kwargs)
    axe.plot(np.arange(-1000, 1001, step=10), hist2, label=title2, **kwargs)
    axe.plot(np.arange(-1000, 1001, step=10), hist3, label=title3, **kwargs)
    axe.plot(np.arange(-1000, 1001, step=10), hist4, label=title4, **kwargs)

    fig.tight_layout()
    plt.legend()
    plt.show()


def compute_mask(file_ref, field_value):
    """Compute mask with a threshold value."""

    ds_ref = rio.open(file_ref)
    im_ref = ds_ref.read(1)
    valid_ref = im_ref != ds_ref.nodata
    mask_ref = im_ref == field_value
    del im_ref, ds_ref

    return mask_ref, valid_ref


def compute_ndxi(im_b1, im_b2, valid):
    """Compute Normalize Difference X Index.
    Rescale to [-1000, 1000] int16 with nodata value = 32767
    1000 * (im_b1 - im_b2) / (im_b1 + im_b2)
    """

    np.seterr(divide="ignore", invalid="ignore")

    print("Compute NDI ...", end="")
    start_time = time.time()
    im_ndxi = 1000.0 - (2000.0 * np.float32(im_b2)) / (
        np.float32(im_b1) + np.float32(im_b2)
    )
    im_ndxi[np.logical_or(im_ndxi < -1000.0, im_ndxi > 1000.0)] = np.nan
    im_ndxi[np.logical_not(valid)] = np.nan
    valid_ndxi = np.isfinite(im_ndxi)
    np.nan_to_num(im_ndxi, copy=False, nan=32767)
    im_ndxi = np.int16(im_ndxi)
    print("in", time.time() - start_time, "seconds.")

    return im_ndxi, valid_ndxi


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


def build_stack(args):
    """Build stack."""

    # Band positions in PHR image
    if args.red == 1:
        names_stack = ["R", "G", "B", "NIR"]
    else:
        names_stack = ["B", "G", "R", "NIR"]

    # Image PHR (numpy array, 4 bands, band number is first dimension),
    ds_phr = rio.open(args.file_phr)
    print_dataset_infos(ds_phr, "PHR")
    im_phr = ds_phr.read()
    nodata_phr = ds_phr.nodata

    # Valid_phr (boolean numpy array, True = valid data, False = no data)
    valid_phr = np.logical_and.reduce(im_phr != nodata_phr, axis=0)
    io_utils.save_image(
        valid_phr.astype(np.uint8),
        join(dirname(args.file_classif), "valid.tif"),
        ds_phr.crs,
        ds_phr.transform,
        nodata=None,
        rpc=ds_phr.tags(ns="RPC"),
    )

    # Save crs, transform and rpc in args
    args.shape = ds_phr.shape
    args.crs = ds_phr.crs
    args.transform = ds_phr.transform
    args.rpc = ds_phr.tags(ns="RPC")

    # Compute NDVI
    if args.file_ndvi:
        ds_ndvi = rio.open(args.file_ndvi)
        print_dataset_infos(ds_ndvi, "NDVI")
        im_ndvi = ds_ndvi.read(1)
        valid_ndvi = im_ndvi != ds_ndvi.nodata
        del ds_ndvi
    else:
        im_ndvi, valid_ndvi = compute_ndxi(
            im_phr[names_stack.index("NIR")],
            im_phr[names_stack.index("R")],
            valid_phr,
        )
        io_utils.save_image(
            im_ndvi,
            join(dirname(args.file_classif), "ndvi.tif"),
            ds_phr.crs,
            ds_phr.transform,
            nodata=32767,
            rpc=ds_phr.tags(ns="RPC"),
        )

    # Compute NDWI
    if args.file_ndwi:
        ds_ndwi = rio.open(args.file_ndwi)
        print_dataset_infos(ds_ndwi, "NDWI")
        im_ndwi = ds_ndwi.read(1)
        valid_ndwi = im_ndwi != ds_ndwi.nodata
        del ds_ndwi
    else:
        im_ndwi, valid_ndwi = compute_ndxi(
            im_phr[names_stack.index("G")],
            im_phr[names_stack.index("NIR")],
            valid_phr,
        )
        io_utils.save_image(
            im_ndwi,
            join(dirname(args.file_classif), "ndwi.tif"),
            ds_phr.crs,
            ds_phr.transform,
            nodata=32767,
            rpc=ds_phr.tags(ns="RPC"),
        )

    # Show NDVI and NDVI images
    if args.display:
        show_images(im_ndvi, "NDVI", im_ndwi, "NDWI", vmin=-1000, vmax=1000)
        show_histograms(im_ndvi, "NDVI", im_ndwi, "NDWI")

    # Global mask construction
    valid_stack = np.logical_and.reduce((valid_phr, valid_ndvi, valid_ndwi))
    del valid_ndvi, valid_ndwi

    # Show PHR and stack validity masks
    if args.display:
        show_images(valid_phr, "Valid PHR", valid_stack, "Valid Stack")

    # Stack construction
    start_time = time.time()
    if args.use_rgb_layers:
        im_stack = np.stack(
            (
                *im_phr,
                im_ndvi,
                im_ndwi,
                *(
                    rio.open(file_layer).read(1)
                    for file_layer in args.files_layers
                ),
            )
        )
    else:
        names_stack = ["NIR"]
        im_stack = np.stack(
            (
                im_phr[names_stack.index("NIR")],
                im_ndvi,
                im_ndwi,
                *(
                    rio.open(file_layer).read(1)
                    for file_layer in args.files_layers
                ),
            )
        )

    names_stack.extend(["NDVI", "NDWI"])
    names_stack.extend(args.files_layers)
    print(names_stack)
    print("Stack time :", time.time() - start_time)
    print("Stack shape :", im_stack.shape)

    return im_stack, valid_stack, names_stack


def build_samples(im_stack, valid_stack, args):
    """Build samples."""
    print("Build samples")
    start_time = time.time()

    ds_gt = rio.open(args.urban_raster)
    im_gt = ds_gt.read(1)
    mask_building = im_gt == 1

    # Check building mask
    #    if np.count_nonzero(mask_building) < 2000:
    #       print('=> Warning : low water pixel number in Pekel Mask\n')

    # Prepare water and other samples

    # If samples not fix:

    nb_building_samples = 2000
    nb_other_samples = 2000

    # Building samples
    rows_b, cols_b = get_indexes_from_masks(
        nb_building_samples, im_gt, args.value_classif, valid_stack,args
    )
    rows_road, cols_road = get_indexes_from_masks(
        nb_building_samples, im_gt, 2, valid_stack, args
    )

    rows_nob = []
    cols_nob = []

    rows_nob, cols_nob = get_indexes_from_masks(
        nb_other_samples, im_gt, 0, valid_stack, args
    )


    save_indexes(
        join(dirname(args.file_classif), "samples_building.tif"),
        zip(rows_b, cols_b),
        zip(rows_nob, cols_nob),
        args.shape,
        args.crs,
        args.transform,
        args.rpc,
    )
    save_indexes(
        join(dirname(args.file_classif), "samples_road.tif"),
        zip(rows_road, cols_road),
        zip(rows_nob, cols_nob),
        args.shape,
        args.crs,
        args.transform,
        args.rpc,
    )

    # samples = building+non building
    rows = rows_b + rows_nob + rows_road
    cols = cols_b + cols_nob + cols_road

    x_samples = np.transpose(im_stack[:, rows, cols])
    y_samples = im_gt[rows, cols]
    del im_gt, ds_gt
    print("Build samples time : ", time.time() - start_time)

    return x_samples, y_samples, mask_building


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


def predict(classifier, im_stack, valid_stack):
    """Predict."""

    start_time = time.time()
    im_predict = np.zeros(im_stack[0].shape, dtype=np.uint8)
    im_proba = np.zeros(
        (2, im_stack[0].shape[0], im_stack[0].shape[1]), dtype=np.uint8
    )
    print("DBG >> Prediction ")
    im_predict[valid_stack] = classifier.predict(
        np.transpose(im_stack[:, valid_stack])
    )
    print(" len data " + str(len(np.transpose(im_stack[:, valid_stack]))))
    proba = classifier.predict_proba(np.transpose(im_stack[:, valid_stack]))
    # print("Shape : "+str(proba.shape()))
    print(im_proba.shape)
    # im_proba = proba.reshape(2, im_stack[0].shape[0], im_stack[0].shape[1])
    # im_proba[0,:,:] = 100*proba[:,0].reshape(im_stack[0].shape[0], im_stack[0].shape[1])
    # im_proba[1,:,:] = 100*proba[:,1].reshape(im_stack[0].shape[0], im_stack[0].shape[1])

    # im_proba[valid_stack] = [0]
    print("Prediction time :", time.time() - start_time)

    return im_predict, im_proba


def classify(args):
    """Compute water mask of file_phr with help of Pekel and Hand images."""

    # Build stack with all layers
    im_stack, valid_stack, names_stack = build_stack(args)
    if args.model:

        classifier = joblib.load(args.model)
    # Create and train classifier from samples
    else:
        x_samples, y_samples, mask_building = build_samples(
        im_stack, valid_stack, args
        )
        classifier = RandomForestClassifier(
            n_estimators=100, max_depth=3, random_state=0, n_jobs=4
        )
        print("RandomForest parameters:\n", classifier.get_params(), "\n")
        train_classifier(classifier, x_samples, y_samples)
        print("Dump classifier to model_rf.dump")

        joblib.dump(
            classifier, join(dirname(args.file_classif), "model_rf.dump")
        )
    print("RandomForest parameters:\n", classifier.get_params(), "\n")

    print("Dump classifier to model_rf.dump")
    joblib.dump(classifier, "model_rf.dump")
    # show_rftree(classifier.estimators_[5], names_stack)
    print_feature_importance(classifier, names_stack)
    gc.collect()

    # Predict and filter with Hand
    im_predict, im_proba = predict(classifier, im_stack, valid_stack)
    print(">> DEBUG >> prediction OK")

    crs, transform, rpc = get_crs_transform(args.file_phr)
    io_utils.save_image(
        im_predict,
        args.file_classif,
        crs,
        transform,
        255,
        rpc,
    )

    io_utils.save_image_2_bands(
        im_proba,
        join(dirname(args.file_classif), "proba.tif"),
        crs,
        transform,
        255,
        rpc,
    )

    #    save_image(im_classif, args.file_classif,
    #               crs, transform, 255, rpc)

    # Show output images
    if args.display:
        show_images(
            im_predict,
            "Predict image",
            mask_building,
            "Mask building",
            vmin=0,
            vmax=1,
        )
        # show_images(mask_pekel, 'Pekel mask', im_classif, 'Classif image', vmin=0, vmax=1)


def getarguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Compute Water Mask.")

    parser.add_argument("file_phr", help="PHR filename")

    parser.add_argument("-red", default=1, help="Red band index")

    parser.add_argument(
        "-display",
        required=False,
        action="store_true",
        help="Display images while running",
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
        "-use_rgb_layers",
        default=False,
        required=False,
        action="store_true",
        dest="use_rgb_layers",
        help="Add R,G,B layers to image stack for learning",
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
        required=True,
        action="store",
        dest="urban_raster",
    )

    parser.add_argument(
        "file_classif",
        help="Output classification filename (default is classif.tif)",
    )

    parser.add_argument(
        "-value_classif",
        type=int,
        default=1,
        required=False,
        action="store",
        dest="value_classif",
        help="Output classification value (default is 1)",
    )

    parser.add_argument(
        "-nb_samples",
        type=int,
        default=2000,
        required=False,
        action="store",
        dest="nb_samples",
        help="Number of samples for the class of interest",
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
        "-model",
        default=None,
        required=False,
        action="store",
        dest="model",
        help="Filepath model",
    )


    return parser.parse_args()



def main():
    try:
        arguments = getarguments()
        classify(arguments)

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
