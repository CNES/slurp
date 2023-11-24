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
        "/datalake/static_aux/MASQUES/WSF/WSF2019_v1/WSF2019_v1.vrt",
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
    nodata_phr = ds_phr.nodata
    
    # Save crs, transform and rpc in args
    args.shape = ds_phr.shape
    args.crs = ds_phr.crs
    args.transform = ds_phr.transform
    args.rpc = ds_phr.tags(ns="RPC")
    
    # Read image
    im_phr = ds_phr.read()
    ds_phr.close()
    del ds_phr
    
    if not np.issubdtype(im_phr.dtype, np.integer):
        raise Exception("The input image must have an integer (signed or unsigned) type but is in " + str(im_phr.dtype))     

    # Valid_phr (boolean numpy array, True = valid data, False = no data)
    valid_phr = np.logical_and.reduce(im_phr != nodata_phr, axis=0)
    if args.save_mode == "debug":
        io_utils.save_image(
            valid_phr.astype(np.uint8),
            join(dirname(args.file_classif), "valid.tif"),
            args.crs,
            args.transform,
            nodata=None,
            rpc=args.rpc,
        )

    # Compute NDVI
    if args.file_ndvi:
        ds_ndvi = rio.open(args.file_ndvi)
        print_dataset_infos(ds_ndvi, "NDVI")
        im_ndvi = ds_ndvi.read(1)
        ndvi_nodata = ds_ndvi.nodata
        ds_ndvi.close()
        del ds_ndvi
        valid_ndvi = im_ndvi != ndvi_nodata
    else:
        im_ndvi, valid_ndvi = compute_ndxi(
            im_phr[names_stack.index("NIR")],
            im_phr[names_stack.index("R")],
            valid_phr,
        )
        if (args.save_mode != "none" and args.save_mode != "aux"):
            io_utils.save_image(
                im_ndvi,
                join(dirname(args.file_classif), "ndvi.tif"),
                args.crs,
                args.transform,
                nodata=32767,
                rpc=args.rpc,
            )

    valid_watermask = np.zeros(valid_phr.shape, dtype=np.uint8)
    if args.file_watermask:
        ds_wm = rio.open(args.file_watermask)
        watermask = ds_wm.read(1)
        ds_wm.close()
        valid_watermask = watermask != 1
        del watermask, ds_wm
        
    valid_vegmask = np.zeros(valid_phr.shape, dtype=np.uint8)
    if args.file_vegetationmask:
        ds_vegmask = rio.open(args.file_vegetationmask)
        vegmask = ds_vegmask.read(1)
        ds_vegmask.close()
        valid_vegmask = vegmask < args.vegmask_max_value
        del vegmask, ds_vegmask
    
    # Compute NDWI
    if args.file_ndwi:
        ds_ndwi = rio.open(args.file_ndwi)
        print_dataset_infos(ds_ndwi, "NDWI")
        im_ndwi = ds_ndwi.read(1)
        ndwi_nodata = ds_ndwi.nodata
        ds_ndwi.close()
        del ds_ndwi
        valid_ndwi = im_ndwi != ndwi_nodata
    else:
        im_ndwi, valid_ndwi = compute_ndxi(
            im_phr[names_stack.index("G")],
            im_phr[names_stack.index("NIR")],
            valid_phr,
        )
        if (args.save_mode != "none" and args.save_mode != "aux"):
            io_utils.save_image(
                im_ndwi,
                join(dirname(args.file_classif), "ndwi.tif"),
                args.crs,
                args.transform,
                nodata=32767,
                rpc=args.rpc,
            )

    # Show NDVI and NDVI images
    if args.display:
        show_images(im_ndvi, "NDVI", im_ndwi, "NDWI", vmin=-1000, vmax=1000)
        show_histograms(im_ndvi, "NDVI", im_ndwi, "NDWI")

    # Global mask construction
    valid_stack = np.logical_and.reduce((valid_phr, valid_ndvi, valid_ndwi))
    if args.file_watermask:
        valid_stack = np.logical_and.reduce((valid_stack, valid_watermask))
    if args.file_vegetationmask:
        valid_stack = np.logical_and.reduce((valid_stack, valid_vegmask))
    del valid_ndvi, valid_ndwi

    # Show PHR and stack validity masks
    if args.display:
        show_images(valid_phr, "Valid PHR", valid_stack, "Valid Stack")     
    
    # Stack construction in a shared memory :
    # 0 -> bands_phr : im_phr (1 or 4 bands)
    # bands_phr : ndvi
    # bands_phr + 1 : ndwi
    # bands_phr + 2 -> bands_phr + 2 + len(file_layers) : file_layers (1 band for each layer)
    # -5 : mask_building
    # -4 : valid_stack
    # -3 and -2 : proba (2 bands)
    # -1 : predict -> index -1
    start_time = time.time()
    bands_phr = im_phr.shape[0] if args.use_rgb_layers else 1
    shm_shape = (bands_phr + 7 + len(args.files_layers), im_phr.shape[1], im_phr.shape[2])
    shm_dtype = np.dtype(np.int16)
    d_size = shm_dtype.itemsize * np.prod(shm_shape)
    shm_key = str(uuid.uuid4())
    shmSlumIn = shared_memory.SharedMemory(create=True, size=d_size, name=shm_key)
    shmNpArray_stack = np.ndarray(shape=shm_shape, dtype=shm_dtype, buffer=shmSlumIn.buf)
    
    #Add im_phr
    if args.use_rgb_layers:
        np.copyto(shmNpArray_stack[:bands_phr, :, :], im_phr.astype(shm_dtype))
    else:
        np.copyto(shmNpArray_stack[:bands_phr, :, :], im_phr[names_stack.index("NIR")].astype(shm_dtype))
        names_stack = ["NIR"]
    del im_phr
    
    #Add im_ndvi and im_ndwi
    np.copyto(shmNpArray_stack[bands_phr, :, :], im_ndvi)
    np.copyto(shmNpArray_stack[bands_phr+1, :, :], im_ndwi)
    del im_ndvi, im_ndwi
    
    #Add files_layers
    for i in range(len(args.files_layers)):
        file_layer = args.files_layers[i]
        ds_layer = rio.open(file_layer)
        layer = ds_layer.read(1)
        ds_layer.close()
        del ds_layer
        np.copyto(shmNpArray_stack[bands_phr+2+i, :, :], layer.astype(shm_dtype))
        
    #Add valid_stack
    shmNpArray_stack[-4, :, :] = valid_stack[:, :]
    del valid_stack
          
    shmSlumIn.close()
    
    names_stack.extend(["NDVI", "NDWI"])
    names_stack.extend(args.files_layers)
    print(names_stack)
    print("Stack time :", time.time() - start_time)
    print("Stack shape :", shm_shape)

    return shm_key, shm_shape, shm_dtype, names_stack


def build_samples(shm_key, shm_shape, shm_dtype, args):
    """Build samples."""
    print("Build samples")
    start_time = time.time()
    shm = shared_memory.SharedMemory(name=shm_key)
    shmNpArray_stack = np.ndarray(shm_shape, dtype=shm_dtype,buffer=shm.buf)
    valid_stack = shmNpArray_stack[-4, :, :]
    
    if args.urban_raster:
        ds_gt = rio.open(args.urban_raster)
        im_gt = ds_gt.read(1)
        ds_gt.close()
        del ds_gt
    else:
        write = False if (args.save_mode == "none" or args.save_mode == "prim") else True
        args.urban_raster = join(dirname(args.file_classif), "wsf.tif")
        im_gt = wsf_recovery(args.file_phr, args.urban_raster, write)
        
    mask_building = im_gt == args.value_classif
        
    nb_building_samples = args.nb_samples
    nb_other_samples = args.nb_samples

    # Building samples
    rows_b, cols_b = get_indexes_from_masks(
        nb_building_samples, im_gt, args.value_classif, valid_stack,args
    )
    
    rows_road = []
    cols_road = []

    if args.nb_classes == 2:
        rows_road, cols_road = get_indexes_from_masks(
            nb_building_samples, im_gt, 2, valid_stack, args
        )
    
    rows_nob = []
    cols_nob = []

    rows_nob, cols_nob = get_indexes_from_masks(
        nb_other_samples, im_gt, 0, valid_stack, args
    )

    if args.save_mode == "debug":
        save_indexes(
            join(dirname(args.file_classif), "samples_building.tif"),
            zip(rows_b, cols_b),
            zip(rows_nob, cols_nob),
            args.shape,
            args.crs,
            args.transform,
            args.rpc,
        )
        if args.nb_classes == 2:
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
    rows = rows_b + rows_nob
    cols = cols_b + cols_nob
    if args.nb_classes == 2:
        rows = rows + rows_road
        cols = cols + cols_road

    im_stack = shmNpArray_stack[:-5, :, :] # PHR image + ndvi + ndwi + file layers
    x_samples = np.transpose(im_stack[:, rows, cols])
    y_samples = im_gt[rows, cols]
    
    # Add mask_building in shared memory
    shmNpArray_stack[-5, :, :] = mask_building[:,:] 
    
    shm.close()
    
    del im_gt
    print("Build samples time : ", time.time() - start_time)

    return x_samples, y_samples


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


def predict_shared(args, classifier, key, im_shape, im_dtype, index, step, lines):
    start_time = time.time()
    shm = shared_memory.SharedMemory(name=key)
    shmNpArray_stack = np.ndarray(im_shape, dtype=im_dtype,buffer=shm.buf)
    im_stack_buffer = shmNpArray_stack[:-5, :, index*step:min((index+1)*step, lines)] # PHR image + ndvi + ndwi + file layers
    valid_stack_buffer = shmNpArray_stack[-4, :, index*step:min((index+1)*step, lines)]
    chunkBuffer =  np.transpose(im_stack_buffer[:, valid_stack_buffer.astype(np.bool_)])
    shm.close()
    del shm
    
    # Probabilities
    proba = classifier.predict_proba(chunkBuffer)
    del chunkBuffer
    
    # Prediction, inspired by sklearn code to predict class
    res_classif = classifier.classes_.take(np.argmax(proba, axis=1), axis=0)
    res_classif[res_classif == args.value_classif] = 1
    
    # Add in stack construction
    shm = shared_memory.SharedMemory(name=key)
    shmNpArray_stack = np.ndarray(im_shape, dtype=im_dtype,buffer=shm.buf)
    valid_stack_buffer = np.copy(shmNpArray_stack[-4, :, index*step:min((index+1)*step, lines)]).astype(np.bool_)
    
    proba_buffer = shmNpArray_stack[-3:-1, :, index*step:min((index+1)*step, lines)]
    proba_buffer[:, valid_stack_buffer] = 100*np.transpose(proba)
    shmNpArray_stack[-3:-1, :, index*step:min((index+1)*step, lines)] = proba_buffer[:, :, :]
    del proba, proba_buffer
    
    predict_buffer = shmNpArray_stack[-1, :, index*step:min((index+1)*step, lines)]
    predict_buffer[valid_stack_buffer] = res_classif   
    shmNpArray_stack[-1, :, index*step:min((index+1)*step, lines)] = predict_buffer[:, :]
    del res_classif; predict_buffer
    
    shm.close()
    
    return index, time.time()-start_time


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


def regul_shared(args, key, im_shape, im_dtype, index, step):
    start_time = time.time()
    start_with_margin, end_with_margin, start_extract, end_extract = get_bornes(index, step, im_shape[2], args.margin)
    # start_with_margin and end_with_margin : indexes to extract the tile with margin from im_predict
    # start_extract and end_extract : indexes to extract the tile without margin from predictBuffer 
    
    shm = shared_memory.SharedMemory(name=key)
    shmNpArray_stack = np.ndarray(im_shape, dtype=im_dtype,buffer=shm.buf)
    predictBuffer = np.copy(shmNpArray_stack[-1, :, start_with_margin:end_with_margin]) # Extract of prediction with margin
    shm.close()
    del shm
   
    # Clean
    im_classif = clean(args, predictBuffer)
    
    # Watershed regulation
    im_seg, gradients, markers = watershed_regul(args, im_classif, key, im_shape, im_dtype, start_with_margin, end_with_margin)
    
    return index, im_classif[:, start_extract:end_extract], im_seg[:, start_extract:end_extract], gradients[:, start_extract:end_extract], markers[:, start_extract:end_extract], time.time()-start_time


def watershed_regul(args, clean_predict, key, im_shape, im_dtype, start_with_margin, end_with_margin):    
    shm = shared_memory.SharedMemory(name=key)
    shmNpArray_stack = np.ndarray(im_shape, dtype=im_dtype,buffer=shm.buf)
    
    # Compute gradient : either on NDVI image, or on RGB image 
    im_stack = shmNpArray_stack[:-5, :, start_with_margin:end_with_margin] # PHR image + ndvi + ndwi + file layers
    if args.use_rgb_layers:
        im_mono = 0.29*im_stack[0] + 0.58*im_stack[1] + 0.114*im_stack[2]
    else:
        im_mono = im_stack[0]
    
    # compute gradient
    edges = sobel(im_mono)
    del im_mono

    # markers map : -1, 1 and 2 : probable background, buildings or false positive
    proba = shmNpArray_stack[-2, :, start_with_margin:end_with_margin] # proba of building class
    markers = np.zeros_like(im_stack[0])       
    probable_buildings = np.logical_and(proba > args.confidence_threshold, clean_predict == 1)
    probable_background = np.logical_and(proba < 20, clean_predict == 0)
    markers[probable_background] = -1
    markers[probable_buildings] = 1    
    del probable_buildings, probable_background        

    if args.remove_false_positive:
        ground_truth = shmNpArray_stack[-5, :, start_with_margin:end_with_margin]
        # mark as false positive pixels with high confidence but not covered by dilated ground truth
        false_positive = np.logical_and(binary_dilation(ground_truth, disk(20)) == 0, proba > args.confidence_threshold)
        markers[false_positive] = 2
        del ground_truth, false_positive
        
    shm.close()
    del shm
    
    # watershed segmentation
    seg = segmentation.watershed(edges, markers)
    seg[seg==-1] = 0
    
    # remove small artefacts 
    if args.remove_small_objects:
        res = remove_small_objects(seg.astype(bool), args.remove_small_objects, connectivity=2).astype(np.uint8)
        # res is either 0 or 1 : we multiply by seg to keep 0/1/2 classes
        seg = np.multiply(res, seg)
    
    return seg, edges, markers


def get_step(args, current_mem, shm_shape, lines):
    base_gb = 225/1024  # Gb
    mem_used_gb = current_mem / 1024  # Gb
    
    # Margin calculation
    if not args.margin:
        args.margin = max(1, 2*args.binary_opening, 2*args.binary_closing, ceil(sqrt(args.remove_small_objects)))
    
    #Max step for prediction
    weight_one_chunk = (shm_shape[0] - 5) * np.dtype(np.int16).itemsize * shm_shape[1] / (1024*1024*1024)
    weight_one_proba = 2 * np.dtype(np.float64).itemsize * shm_shape[1] / (1024*1024*1024)  # Memory for one line in Gb    
    max_step_predict = (((args.max_memory - mem_used_gb) / args.nb_workers) - base_gb) / (7*weight_one_chunk + weight_one_proba)  # float
    
    if max_step_predict < 1:
        raise Exception("Insufficient memory, you need to increase the max memory")
    
    #Max step for regularization
    weight_one_clean = np.dtype(np.int16).itemsize * shm_shape[1] / (1024*1024*1024)
    weight_one_segment = np.dtype(np.int32).itemsize * shm_shape[1] / (1024*1024*1024)
    weight_one_grad = np.dtype(np.float64).itemsize * shm_shape[1] / (1024*1024*1024)
    weight_one_marks = np.dtype(np.int16).itemsize * shm_shape[1] / (1024*1024*1024)  # Memory for one line in Gb
    memory_regul = weight_one_clean + weight_one_segment + weight_one_marks + 9*weight_one_grad
    max_step_regul = (((args.max_memory - mem_used_gb) / args.nb_workers) - base_gb - memory_regul*args.margin) / memory_regul  # float
    
    if max_step_regul < 1:
        raise Exception("Insufficient memory, you need to increase the max memory")

    step = min(int(max_step_predict), int(max_step_regul), lines // args.nb_workers + 1)
    
    #print("Prediction max memory use (in Gb)", mem_used_gb + args.nb_workers * (base_gb + (7*weight_one_chunk + weight_one_proba) * step))
    #print("Regularization max memory use (in Gb)", mem_used_gb + args.nb_workers * (base_gb + memory_regul * (step + args.margin)))
                                                                     
    return step
   

def predict(args, classifier, shm_key, shm_shape, shm_dtype, current_mem):
    """Predict."""    
    lines = shm_shape[2]
    step = get_step(args, current_mem, shm_shape, lines)  # number of lines of each tile
    nb_tiles = lines//step if (lines % step == 0) else lines//step + 1
    print("Division in " + str(nb_tiles) + " tiles")
    
    print("DBG >> Prediction ")
    start_time = time.time()
    
    context = get_context('spawn')    
    future_seg = []
    cpt = 0
    
    time_predict = 0
    t0_predict = time.time()
    
    # Prediction
    while (cpt < nb_tiles):
        endSession = min(cpt + args.nb_workers, nb_tiles)
        workers = endSession - cpt
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            for index in range(cpt, endSession):
                future_seg.append(executor.submit(predict_shared, args, classifier, shm_key, shm_shape, shm_dtype, index, step, lines))
                
        for seg in concurrent.futures.as_completed(future_seg):
            try:
                num, time_exec = seg.result()
                time_predict += time_exec
            except Exception as e:
                print("Exception ---> "+str(e))
        
        cpt += args.nb_workers
        future_seg = []
    
    time_predict_user = time.time() - t0_predict
    
    print("Prediction time :", time.time() - start_time)
    
    print("DBG >> Regularization ")
    t0 = time.time()
    
    im_clean = np.zeros(shm_shape[1:], dtype=np.int16)
    segment = np.zeros(shm_shape[1:], dtype=np.int32)
    grad = np.zeros(shm_shape[1:], dtype=np.float64)
    marks = np.zeros(shm_shape[1:], dtype=np.int16)

    context = get_context('spawn')
    future_seg = []
    cpt = 0
    
    time_regul = 0
    t0_regul = time.time()

    # Regularization
    while (cpt < nb_tiles):
        endSession = min(cpt + args.nb_workers, nb_tiles)
        workers = endSession - cpt
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            for index in range(cpt, endSession):
                future_seg.append(executor.submit(regul_shared, args, shm_key, shm_shape, shm_dtype, index, step))
                
        for seg in concurrent.futures.as_completed(future_seg):
            try:
                num, res_clean, res_seg, res_gradients, res_markers, time_exec = seg.result()
                np.copyto(im_clean[:, num*step:min((num+1)*step, shm_shape[2])], res_clean)
                np.copyto(segment[:, num*step:min((num+1)*step, shm_shape[2])], res_seg)
                np.copyto(grad[:, num*step:min((num+1)*step, shm_shape[2])], res_gradients)
                np.copyto(marks[:, num*step:min((num+1)*step, shm_shape[2])], res_markers)
                time_regul += time_exec
            except Exception as e:
                print("Exception ---> "+str(e)) 
        
        cpt += args.nb_workers
        future_seg = []
        
    time_regul_user = time.time() - t0_regul
    
    print(">> Regularization in %d sec " % (time.time()-t0))

    return im_clean, segment, grad, marks, [time_predict_user, time_predict, time_regul_user, time_regul]


def classify(args):
    """Compute water mask of file_phr with help of Pekel and Hand images."""
    t0 = time.time()

    # Build stack with all layers
    shm_key, shm_shape, shm_dtype, names_stack = build_stack(args)
    
    time_stack = time.time()
    
    # Create and train classifier from samples
    x_samples, y_samples = build_samples(shm_key, shm_shape, shm_dtype, args)
    
    time_samples = time.time()
        
    classifier = RandomForestClassifier(
        n_estimators=args.nb_estimators, max_depth=args.max_depth, class_weight="balanced",
        random_state=0, n_jobs=args.nb_jobs
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
    
    #Memory used
    base = 215 # poids imports
    suivi_mem = base + (x_samples.nbytes + y_samples.nbytes) / (1024*1024)

    # Predict and filter with Hand
    im_classif, im_seg, gradients, markers, all_times = predict(args, classifier, shm_key, shm_shape, shm_dtype, suivi_mem)
    print(">> DEBUG >> prediction OK")
    
    time_random_forest = time.time()

    #Save outputs
    crs, transform, rpc = get_crs_transform(args.file_phr)
    
    shm = shared_memory.SharedMemory(name=shm_key)
    shmNpArray_stack = np.ndarray(shm_shape, dtype=shm_dtype,buffer=shm.buf)
    im_predict = shmNpArray_stack[-1, :, :]
    
    # Raw prediction
    io_utils.save_image(
        im_predict,
        args.file_classif,
        crs,
        transform,
        255,
        rpc,
    )
    
    # Final classification (cleaned + regularized) 
    io_utils.save_image(
        im_seg,
        args.file_classif.replace(".tif","_seg.tif"),
        crs,
        transform,
        255,
        rpc,
    )
    
    if args.save_mode == "debug":
        # Clean classification (morphological operations)
        io_utils.save_image(
            im_classif,
            args.file_classif.replace(".tif","_clean.tif"),
            crs,
            transform,
            255,
            rpc,
        )
        # Gradient image (from mono-band image)
        gradients = 1000*gradients
        io_utils.save_image(
            gradients,
            args.file_classif.replace(".tif","_gradient.tif"),
            crs,
            transform,
            255,
            rpc,
        )
        # Markers image
        io_utils.save_image(
            markers,
            args.file_classif.replace(".tif","_markers.tif"),
            crs,
            transform,
            255,
            rpc,
        )
        # RandomForest confidence image n_classes x [0-100]
        im_proba = shmNpArray_stack[-3:-1, :, :]
        io_utils.save_image_n_bands(
            im_proba,
            join(dirname(args.file_classif), basename(args.file_classif).replace(".tif", "_proba.tif")),
            crs,
            transform,
            255,
            rpc,
        )
   
    # Show output images
    if args.display:
        mask_building = shmNpArray_stack[-5, :, :]
        show_images(
            im_predict,
            "Predict image",
            mask_building,
            "Mask building",
            vmin=0,
            vmax=1,
        )
        # show_images(mask_pekel, 'Pekel mask', im_classif, 'Classif image', vmin=0, vmax=1)
        
    shm.close()
    shm.unlink()
    
    end_time = time.time()
        
    print("**** Urban mask for "+str(args.file_phr)+" (saved as "+str(args.file_classif)+") ****")
    print("Total time (user)       :\t"+convert_time(end_time-t0))
    print("- Build_stack           :\t"+convert_time(time_stack-t0))
    print("- Build_samples         :\t"+convert_time(time_samples-time_stack))
    print("- Random forest (total) :\t"+convert_time(time_random_forest-time_samples))
    print("- Post-processing       :\t"+convert_time(end_time-time_random_forest))
    print("***")
    print("Max workers used for parallel tasks "+str(args.nb_workers))
    print("Prediction (user)       :\t"+convert_time(all_times[0]))
    print("Prediction (parallel)   :\t"+convert_time(all_times[1]))
    print("Regularization (user)   :\t"+convert_time(all_times[2]))
    print("Regularization (parallel):\t"+convert_time(all_times[3]))
    
    
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

    #print(">> Clean in %d sec " % (time.time()-t0))

    return im_classif


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
        "-nb_samples",
        type=int,
        default=1000,
        required=False,
        action="store",
        dest="nb_samples",
        help="Number of samples for the class of interest",
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
