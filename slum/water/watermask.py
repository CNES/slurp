#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Compute water mask of PHR image with help of Pekel and Hand images."""

# Code linted with python linter pylint, score > 9

# XR: TODO: Utiliser les cartes de transitions
# XR: TODO: Traiter le cas hand_strict=True + hand_filter=True
# XR: TODO: Utiliser les cartes saisonnières
# XR: TODO: Forcer un ratio de max 5 sur le nombre de points d'apprentissage de chaque classe
# XR: TODO: Détecter automatiquement le "biome", dry, water, ...
# XR: TODO: Traiter le cas où il y a très peu de point dans le masque pekel (ajuster le seuil)
# XR: TODO: Pouvoir désactiver le filtre pekel
# XR: DONE: Faire des groupes d'options
# XR: DONE: Renommer samples_water, ... en nb_samples_water, ...
# XR: DONE: Ajouter une option hand_filter et une option pekel_filter
# XR: DONE: Remove dead code
# XR: DONE: Renommer l'option optim1 pour être plus explicite => hand_strict
# XR: DONE: Supprimer option nofilter
# XR: DONE: Ajouter un auto config pour calculer le nombre de samples de chaque classe
# XR: DONE: Ajout option smart_area_pct
# XR: DONE: Ajout choix des échantillons sur une grille
# XR: DONE: Utiliser les dernières versions de Pekel/GSW
# XR: DONE: Ajouter une option de fermeture morphologique
# XR: DONE: Sauvegarde des échantillons dilatés avec colormap
# XR: DONE: Ajout gestion des nuages au format .GML
# XR: DONE: Donner plus d'importance aux grosses zones d'eau dans l'échantillonage (smart2)
# XR: DONE: Choix des points d'échantillonage dans chaque zone d'eau (smart1)
# XR: DONE: Ajout paramètre colormap dans save_image
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
# XR: CANCEL: Utiliser l'attributs rasterio ds.descriptions=("",...) (liste de string)
# XR: DONE: Utiliser rasterio .tags() et .update_tags(ns='namespace', ...)
# XR: DONE: Ajouter des métadonnées dans le masque de sortie (valeur thresh, rgb, ...)
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
import otbApplication as otb
import rasterio as rio
from skimage.filters.rank import maximum
from skimage.measure import label, regionprops
from skimage.morphology import (
    area_closing,
    binary_closing,
    diameter_closing,
    remove_small_holes,
    square,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz, export_text
import concurrent.futures
from multiprocessing import shared_memory, get_context
from pylab import *

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


def save_image(
    image,
    file,
    crs=None,
    transform=None,
    nodata=None,
    rpc=None,
    colormap=None,
    tags=None,
    **kwargs,
):
    """Save 1 band numpy image to file with deflate compression.
    Note that rio.dtype is string so convert np.dtype to string.
    rpc must be a dictionnary.
    """
    
    dataset = rio.open(
        file,
        "w",
        driver="GTiff",
        compress="deflate",
        height=image.shape[0],
        width=image.shape[1],
        count=1,
        dtype=str(image.dtype),
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
        **kwargs,
    )
    axe.plot(
        np.arange(-1000, 1001, step=10),
        hist2,
        color="red",
        label=title2,
        **kwargs,
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


def get_crs_transform_rpc(file):
    """Get CRS, Transform and RPC of a geotiff file."""

    dataset = rio.open(file)
    crs = dataset.crs
    transform = dataset.transform
    rpc = dataset.tags(ns="RPC")
    dataset.close()

    return crs, transform, rpc


def get_grid_indexes_from_masks(im_valid, step):
    """Get valid indexes from masks."""

    rows_idxs = []
    cols_idxs = []

    first_col = 0
    for row in np.arange(0, im_valid.shape[0], step):
        for col in np.arange(first_col, im_valid.shape[1], step):
            if im_valid[row, col]:
                rows_idxs.append(row)
                cols_idxs.append(col)
        first_col = int(step / 2) - first_col

    return rows_idxs, cols_idxs


def get_random_indexes_from_masks(nb_indexes, mask_1, mask_2):
    """Get random valid indexes from masks.
    Mask 1 is a validity mask
    """

    nb_idxs = 0
    rows_idxs = []
    cols_idxs = []

    height = mask_1.shape[0]
    width = mask_1.shape[1]

    nz_rows, nz_cols = np.nonzero(mask_2)
    min_row = np.min(nz_rows)
    max_row = np.max(nz_rows)
    min_col = np.min(nz_cols)
    max_col = np.max(nz_cols)

    while nb_idxs < nb_indexes:
        row = np.random.randint(0, height)
        col = np.random.randint(0, width)

        if mask_1[row, col] and mask_2[row, col]:
            rows_idxs.append(row)
            cols_idxs.append(col)
            nb_idxs += 1

    return rows_idxs, cols_idxs


def get_smart_indexes_from_mask(nb_indexes, pct_area, minimum, mask):

    rows_idxs = []
    cols_idxs = []

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
    save_image(img_dilat, filename, crs, transform, 0, rpc, colormap)

    return


def mask_filter(im_in, mask_ref):
    """Remove water areas in im_in not in contact
    with water areas in mask_ref.
    """

    print("Post process : compute label image...", end="")
    im_label, nb_label = label(im_in, connectivity=2, return_num=True)
    print("found", nb_label, "regions.")

    print("Post process : compute filtered image...", end="")
    start_time = time.time()
    im_label_thresh = np.copy(im_label)
    im_label_thresh[np.logical_not(mask_ref)] = 0
    valid_labels = np.delete(np.unique(im_label_thresh), 0)

    im_filtered = np.zeros(np.shape(mask_ref), dtype=np.uint8)
    im_filtered[np.isin(im_label, valid_labels)] = 1

    print("in", time.time() - start_time, "seconds.")
    print("Post process : keep", np.size(valid_labels), "regions.")

    return im_filtered


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
        save_image(
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
        ds_ndvi.read()  # necessary for a true clean with del
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
            save_image(
                im_ndvi,
                join(dirname(args.file_classif), "ndvi.tif"),
                args.crs,
                args.transform,
                nodata=32767,
                rpc=args.rpc,
            )
    
    print(
        "NDVI : min =",
        np.min(im_ndvi, where=(im_ndvi != 32767), initial=1000),
        "max =",
        np.max(im_ndvi, where=(im_ndvi != 32767), initial=-1000),
    )

    # Compute NDWI
    if args.file_ndwi:
        ds_ndwi = rio.open(args.file_ndwi)
        print_dataset_infos(ds_ndwi, "NDWI")
        ds_ndwi.read()  # necessary for a true clean with del
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
            save_image(
                im_ndwi,
                join(dirname(args.file_classif), "ndwi.tif"),
                args.crs,
                args.transform,
                nodata=32767,
                rpc=args.rpc,
            )
    print(
        "NDWI : min =",
        np.min(im_ndwi, where=(im_ndwi != 32767), initial=1000),
        "max =",
        np.max(im_ndwi, where=(im_ndwi != 32767), initial=-1000),
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
        
    # Stack construction in a shared memory :
    # 0 -> bands_phr : im_phr (1 or 4 bands)
    # bands_phr : ndvi
    # bands_phr + 1 : ndwi
    # bands_phr + 2 -> bands_phr + 2 + len(file_layers) : file_layers (1 band for each layer)
    # -1 : valid_stack
    start_time = time.time()
    bands_phr = im_phr.shape[0] if args.use_rgb_layers else 1
    shm_shape = (bands_phr + 3 + len(args.files_layers), im_phr.shape[1], im_phr.shape[2])
    shm_dtype = np.dtype(np.int16)
    d_size = shm_dtype.itemsize * np.prod(shm_shape)
    shm_key = "slumImStack"
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
    shmNpArray_stack[-1, :, :] = valid_stack[:, :]
    del valid_stack
          
    shmSlumIn.close()
        
    names_stack.extend(["NDVI", "NDWI"])
    names_stack.extend(args.files_layers)
    print("\nStack : ", names_stack)
    print("Stack time :", time.time() - start_time)
    print("Stack shape :", shm_shape)
    print("\n")

    return shm_key, shm_shape, shm_dtype, names_stack


def build_samples(shm_key, shm_shape, shm_dtype, args):
    """Build samples."""
    
    write = False if (args.save_mode == "none" or args.save_mode == "prim") else True
    
    # Image Pekel (numpy array, first band)
    if args.file_pekel:
        ds_ref = rio.open(args.file_pekel)
        print_dataset_infos(ds_ref, "PEKEL")
        pekel_nodata = ds_ref.nodata
        im_pekel = ds_ref.read(1)
        ds_ref.close()
        del ds_ref
    else:
        if 1 <= args.pekel_month <= 12:
            args.file_data_pekel = join(
                dirname(args.file_classif), f"pekel{args.pekel_month}.tif"
            )
            args.file_mask_pekel = join(
                dirname(args.file_classif),
                f"has_observations{args.pekel_month}.tif",
            )
            args.file_pekel = args.file_data_pekel
            im_pekel = pekel_month_recovery(
                args.file_phr,
                args.pekel_month,
                args.file_data_pekel,
                args.file_mask_pekel,
                write,
            )
        else:
            args.file_pekel = join(dirname(args.file_classif), "pekel.tif")
            im_pekel = pekel_recovery(args.file_phr, args.file_pekel, write)
        pekel_nodata = 255.0
    
    # Pekel masks
    if args.hand_strict:
        if args.pekel_filter:
            [mask_pekel, mask_pekelxx, mask_pekel0] = compute_mask(im_pekel, pekel_nodata, [args.thresh_pekel, args.strict_thresh, 0])[0]
        else:
            [mask_pekel, mask_pekelxx] = compute_mask(im_pekel, pekel_nodata, [args.thresh_pekel, args.strict_thresh])[0]
            mask_pekel0 = "not defined"
    elif args.pekel_filter:
        [mask_pekel, mask_pekel0] = compute_mask(im_pekel, pekel_nodata, [args.thresh_pekel, 0])[0]
    else:
        mask_pekel = compute_mask(im_pekel, pekel_nodata, args.thresh_pekel)[0]
        mask_pekel0 = "not defined"
    del im_pekel

    # Check pekel mask
    if np.count_nonzero(mask_pekel) < 2000:
        print("=> Warning : low water pixel number in Pekel Mask\n")        
        shm = shared_memory.SharedMemory(name=shm_key)
        shmNpArray_stack = np.ndarray(shm_shape, dtype=shm_dtype,buffer=shm.buf)
        index_ndwi = shm_shape[0] - 2 - len(args.files_layers) if args.use_rgb_layers else 2
        im_ndvi = np.copy(shmNpArray_stack[index_ndwi])
        shm.close()
        del shm
        mask_pekel = compute_mask(im_ndvi, 32767.0, 0.3)[0]

    # Image HAND (numpy array, first band) and mask
    if args.file_hand:
        ds_hand = rio.open(args.file_hand)
        print_dataset_infos(ds_hand, "HAND")
        hand_nodata = ds_hand.nodata
        im_hand = ds_hand.read(1)
        ds_hand.close()
        del ds_hand
    else:
        args.file_hand = join(dirname(args.file_classif), "hand.tif")
        im_hand = hand_recovery(args.file_phr, args.file_hand, write)
        hand_nodata = -9999.0
    
    # Create HAND mask
    mask_hand = compute_mask(im_hand, hand_nodata, args.thresh_hand)[0]
    del im_hand
    
    # Do not learn in water surface (usefull if image contains big water surfaces)
    # Add some robustness if hand_strict is not used
    if args.hand_strict:
        np.logical_not(np.logical_or(mask_hand, mask_pekelxx), out=mask_hand)
    else:
        np.logical_not(mask_hand, out=mask_hand)

    # Show Pekel and HAND masks
    if args.display:
        show_images(
            mask_pekel,
            "PEKEL > " + str(args.thresh_pekel),
            mask_hand,
            "HAND <" + str(args.thresh_hand),
        )
        
    # Get cloud mask if any
    if args.file_cloud_gml:
        mask_nocloud = np.logical_not(
            cloud_from_gml(args.file_cloud_gml, args.file_phr)
        )
    else:
        mask_nocloud = np.ones(shm_shape[1:], dtype=np.uint8)
    
    shm = shared_memory.SharedMemory(name=shm_key)
    shmNpArray_stack = np.ndarray(shm_shape, dtype=shm_dtype,buffer=shm.buf)
    valid_stack = shmNpArray_stack[-1, :, :]
                
    if args.display:
        show_images(
            mask_nocloud,
            "No Cloud image",
            valid_stack,
            "Validity image",
            vmin=0,
            vmax=1,
        )
    
    # Prepare samples
    valid_samples = np.logical_and(valid_stack, mask_nocloud)
    shm.close()
    del shm, mask_nocloud
    if args.samples_method != "grid":
        # Prepare random water and other samples
        if args.nb_samples_auto:
            nb_valid_pixels = np.count_nonzero(valid_samples)
            nb_valid_water_pixels = np.count_nonzero(
                np.logical_and(mask_pekel, valid_samples)
            )
            nb_valid_other_pixels = nb_valid_pixels - nb_valid_water_pixels
            nb_water_samples = int(nb_valid_water_pixels * args.auto_pct)
            nb_other_samples = int(nb_valid_other_pixels * args.auto_pct)
        else:
            nb_water_samples = args.nb_samples_water
            nb_other_samples = args.nb_samples_other
        print("Use", nb_water_samples, "samples for water")
        print("Use", nb_other_samples, "samples for other")
        
        # Pekel samples
        if args.samples_method == "random":
            rows_pekel, cols_pekel = get_random_indexes_from_masks(
                nb_water_samples, valid_samples, mask_pekel
            )

        if args.samples_method == "smart":
            rows_pekel, cols_pekel = get_smart_indexes_from_mask(
                nb_water_samples,
                args.smart_area_pct,
                args.smart_minimum,
                np.logical_and(mask_pekel, valid_samples),
            )

        # Hand samples, always random (currently)
        rows_hand, cols_hand = get_random_indexes_from_masks(
            nb_other_samples, valid_samples, mask_hand
        )
        
        # All samples
        rows = rows_pekel + rows_hand
        cols = cols_pekel + cols_hand
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

    else:
        # Prepare regular samples
        rows, cols = get_grid_indexes_from_masks(valid_samples, args.grid_spacing)
        if args.save_mode == "debug":
            rc_pekel = [
                (row, col)
                for (row, col) in zip(rows, cols)
                if mask_pekel[row, col] == 1
            ]
            rc_others = [
                (row, col)
                for (row, col) in zip(rows, cols)
                if mask_pekel[row, col] != 1
            ]

            colormap = {
                0: (0, 0, 0, 0),  # nodata
                1: (0, 0, 255),  # eau
                2: (255, 0, 0),  # autre
                3: (0, 0, 0, 0),
            }
            save_indexes(
                "samples.tif",
                rc_pekel,
                rc_others,
                args.shape,
                args.crs,
                args.transform,
                args.rpc,
                colormap,
            )
    
    # Prepare samples for learning
    shm = shared_memory.SharedMemory(name=shm_key)
    shmNpArray_stack = np.ndarray(shm_shape, dtype=shm_dtype,buffer=shm.buf)
    im_stack = shmNpArray_stack[:-1, :, :]
    x_samples = np.transpose(im_stack[:, rows, cols])
    y_samples = mask_pekel[rows, cols]
    shm.close()

    return x_samples, y_samples, [mask_pekel, mask_pekel0], mask_hand


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
    
    
def get_step(args, current_mem, shm_shape, lines):
    base_gb = 215/1024  # Gb
    mem_used_gb = current_mem / 1024  # Gb
    weight_one_chunk = (shm_shape[0] - 1) * np.dtype(np.int16).itemsize * shm_shape[1] / (1024*1024*1024)
    max_step_predict = (((args.max_memory - mem_used_gb) / args.nb_workers) - base_gb) / (2 * (args.nb_jobs + 1) * weight_one_chunk)  # float
    
    if max_step_predict < 1:
        raise Exception("Insufficient memory, you need to increase the max memory")

    step = min(int(max_step_predict), lines // args.nb_workers + 1)
    
    print("Prediction max memory use (in Gb)", mem_used_gb + args.nb_workers * (base_gb + (2 * (args.nb_jobs + 1) * step * weight_one_chunk)))
                                                                     
    return step


def predict_shared(classifier, key, im_shape, im_dtype, index, step, lines):
    start_time = time.time()
    shm = shared_memory.SharedMemory(name=key)
    shmNpArray_stack = np.ndarray(im_shape, dtype=im_dtype,buffer=shm.buf)
    im_stack_buffer = shmNpArray_stack[:-1, :, index*step:min((index+1)*step, lines)] # PHR image + ndvi + ndwi + file layers
    valid_stack_buffer = np.copy(shmNpArray_stack[-1, :, index*step:min((index+1)*step, lines)]).astype(np.bool_)
    chunkBuffer =  np.transpose(im_stack_buffer[:, valid_stack_buffer])
    shm.close()
    del shm
    
    predict_shape = (im_shape[1], min((index+1)*step, lines)-index*step)
    prediction = np.zeros(predict_shape, dtype=np.uint8)    
    prediction[valid_stack_buffer] = classifier.predict(chunkBuffer)
    
    return index, prediction, time.time()-start_time
    
    
def predict(args, classifier, shm_key, shm_shape, shm_dtype, current_mem):
    """Predict."""
    start_time = time.time()
    lines = shm_shape[2]
    step = get_step(args, current_mem, shm_shape, lines)  # number of lines of each tile
    nb_tiles = lines//step if (lines % step == 0) else lines//step + 1
    print("Division in " + str(nb_tiles) + " tiles")
    
    print("DBG >> Prediction ")
    start_time = time.time()
    
    context = get_context('spawn')    
    future_seg = []
    cpt = 0
    
    im_predict = np.zeros(shm_shape[1:], dtype=np.int16)
    
    time_predict = 0
    t0_predict = time.time()
    
    while (cpt < nb_tiles):
        endSession = min(cpt + args.nb_workers, nb_tiles)
        workers = endSession - cpt
        mem = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            for index in range(cpt, endSession):
                future_seg.append(executor.submit(predict_shared, classifier, shm_key, shm_shape, shm_dtype, index, step, lines))
                
        for seg in concurrent.futures.as_completed(future_seg):
            try:
                num, res, time_exec = seg.result()
                np.copyto(im_predict[:, num*step:min((num+1)*step, shm_shape[2])], res)
                time_predict += time_exec
            except Exception as e:
                print("Exception ---> "+str(e))

        cpt += args.nb_workers
        future_seg = []
        
    time_predict_user = time.time() - t0_predict

    print("Prediction time :", time.time() - start_time)
    
    return im_predict, time_predict_user, time_predict


def classify(args):
    """Compute water mask of file_phr with help of Pekel and Hand images."""    
    t0 = time.time()

    # Build stack with all layers
    shm_key, shm_shape, shm_dtype, names_stack = build_stack(args) 
    
    time_stack = time.time()
    
    # Build samples from stack and control layers (pekel, hand)
    x_samples, y_samples, [mask_pekel, mask_pekel0], mask_hand = build_samples(
        shm_key, shm_shape, shm_dtype, args
    )

    time_samples = time.time()

    # Create and train classifier from samples
    classifier = RandomForestClassifier(
        n_estimators=args.nb_estimators, max_depth=args.max_depth, random_state=0, n_jobs=args.nb_jobs
    )
    print("RandomForest parameters:\n", classifier.get_params(), "\n")
    train_classifier(classifier, x_samples, y_samples)
    print("Dump classifier to model_rf.dump")
    joblib.dump(classifier, "model_rf.dump")
    # show_rftree(classifier.estimators_[5], names_stack)
    # show_rftree(classifier.estimators_[10], names_stack)
    # show_rftree(classifier.estimators_[15], names_stack)
    print_feature_importance(classifier, names_stack)
    gc.collect()
    
    #Memory used
    base = 210  # poids imports
    suivi_mem = base + (x_samples.nbytes + y_samples.nbytes + mask_pekel.nbytes + mask_hand.nbytes) / (1024*1024)
    if args.pekel_filter:
        suivi_mem += mask_pekel0.nbytes / (1024*1024)
    
    # Predict and filter with Hand
    im_predict, time_predict_user, time_predict = predict(args, classifier, shm_key, shm_shape, shm_dtype, suivi_mem)
    
    time_random_forest = time.time()
    
    # Filter with Hand
    if args.hand_filter:
        if not args.hand_strict:
            im_predict[np.logical_not(mask_hand)] = 0
        else:
            print("\nWARNING: hand_filter and hand_strict are incompatible.")
    
    # Filter for final classification
    if args.pekel_filter or args.file_esri or (len(args.filter_layers) > 0):
        mask = np.zeros(shm_shape[1:], dtype=bool)
        if args.pekel_filter:  # filter with pekel0
            mask = np.logical_or(mask, mask_pekel0)
        if args.file_esri:  # esri_recovery...
            mask_esri = compute_esri(args.file_esri, "ESRI")[0]
            mask = np.logical_or(mask, mask_esri)
        for i in range(len(args.filter_layers)):  # Other classification files
            filter_layer = args.filter_layers[i]
            ds_layer = rio.open(filter_layer)
            layer = ds_layer.read(1)
            ds_layer.close()
            del ds_layer
            mask = np.logical_or(mask, layer)
                       
        im_classif = mask_filter(im_predict, mask)
    else:
        im_classif = im_predict

    # Closing
    start_time = time.time()
    if args.binary_closing:
        im_classif = binary_closing(
            im_classif, square(args.binary_closing)
        ).astype(np.uint8)
    elif args.diameter_closing:
        # XR: TODO très long voir bloqué
        im_classif = diameter_closing(
            im_classif, args.diameter_closing, connectivity=2
        )
    elif args.area_closing:
        im_classif = area_closing(im_classif, args.area_closing, connectivity=2)
    elif args.remove_small_holes:
        im_classif = remove_small_holes(
            im_classif.astype(bool), args.remove_small_holes, connectivity=2
        ).astype(np.uint8)
    print("Closing time :", time.time() - start_time)
    
    # Add nodata to predict and classif and save (must be done after mask_filter)
    shm = shared_memory.SharedMemory(name=shm_key)
    shmNpArray_stack = np.ndarray(shm_shape, dtype=shm_dtype,buffer=shm.buf)
    valid_stack = shmNpArray_stack[-1, :, :]
    
    im_predict[np.logical_not(valid_stack)] = 255
    im_predict[im_predict == 1] = args.value_classif

    im_classif[np.logical_not(valid_stack)] = 255
    im_classif[im_classif == 1] = args.value_classif
    
    shm.close()
    shm.unlink()

    crs, transform, rpc = get_crs_transform_rpc(args.file_phr)
    save_image(
        im_predict,
        join(dirname(args.file_classif), "predict.tif"),
        crs,
        transform,
        255,
        rpc,
        tags=args.__dict__,
    )
    
    save_image(
        im_classif,
        args.file_classif,
        crs,
        transform,
        255,
        rpc,
        tags=args.__dict__,
    )
    
    # Show output images
    if args.display:
        show_images(
            im_predict,
            "Predict image",
            im_classif,
            "Classif image",
            vmin=0,
            vmax=1,
        )
        show_images(
            mask_pekel,
            "Pekel mask",
            im_classif,
            "Classif image",
            vmin=0,
            vmax=1,
        )

    end_time = time.time()
        
    print("**** Water mask for "+str(args.file_phr)+" (saved as "+str(args.file_classif)+") ****")
    print("Total time (user)       :\t"+convert_time(end_time-t0))
    print("- Build_stack           :\t"+convert_time(time_stack-t0))
    print("- Build_samples         :\t"+convert_time(time_samples-time_stack))
    print("- Random forest (total) :\t"+convert_time(time_random_forest-time_samples))
    print("- Post-processing       :\t"+convert_time(end_time-time_random_forest))
    print("***")
    print("Max workers used for parallel tasks "+str(args.nb_workers))
    print("Prediction (user)       :\t"+convert_time(time_predict_user))
    print("Prediction (parallel)   :\t"+convert_time(time_predict))
    

def convert_time(seconds):
    full_time = time.gmtime(seconds)
    return time.strftime("%H:%M:%S", full_time)


def getarguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Compute Water Mask.")

    group1 = parser.add_argument_group(description="*** INPUT FILES ***")
    group2 = parser.add_argument_group(description="*** OPTIONS ***")
    group3 = parser.add_argument_group(
        description="*** LEARNING SAMPLES SELECTION ***"
    )
    group4 = parser.add_argument_group(description="*** POST PROCESSING ***")
    group5 = parser.add_argument_group(description="*** OUTPUT FILE ***")

    # Input files
    group1.add_argument("file_phr", help="PHR filename")

    group1.add_argument(
        "-pekel",
        default=None,
        required=False,
        action="store",
        dest="file_pekel",
        help="Pekel filename (computed if missing option)",
    )

    group1.add_argument(
        "-thresh_pekel",
        type=float,
        default=50,
        required=False,
        action="store",
        dest="thresh_pekel",
        help="Pekel Threshold float (default is 50)",
    )

    group1.add_argument(
        "-pekel_month",
        type=int,
        default=0,
        required=False,
        action="store",
        dest="pekel_month",
        help="Use monthly recurrence map instead of occurence map",
    )

    group1.add_argument(
        "-hand",
        default=None,
        required=False,
        action="store",
        dest="file_hand",
        help="Hand filename (computed if missing option)",
    )

    group1.add_argument(
        "-thresh_hand",
        type=int,
        default=25,
        required=False,
        action="store",
        dest="thresh_hand",
        help="Hand Threshold int >= 0 (default is 25)",
    )

    group1.add_argument(
        "-ndvi",
        default=None,
        required=False,
        action="store",
        dest="file_ndvi",
        help="NDVI filename (computed if missing option)",
    )

    group1.add_argument(
        "-ndwi",
        default=None,
        required=False,
        action="store",
        dest="file_ndwi",
        help="NDWI filename (computed if missing option)",
    )

    group1.add_argument(
        "-layers",
        nargs="+",
        default=[],
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
        "-esri",
        default=None,
        required=False,
        action="store",
        dest="file_esri",
        help="ESRI filename, will be used in postprocessing",
    )
    
    group1.add_argument(
        "-filters",
        nargs="+",
        default=[],
        required=False,
        action="store",
        dest="filter_layers",
        help="Add filters used in postprocessing",
    )

    # Options
    group2.add_argument("-red", default=1, help="Red band index")

    group2.add_argument(
        "-use_rgb_layers",
        default=False,
        required=False,
        action="store_true",
        dest="use_rgb_layers",
        help="Add R,G,B layers to image stack for learning",
    )

    group2.add_argument(
        "-hand_strict",
        default=False,
        required=False,
        action="store_true",
        dest="hand_strict",
        help="Use not(pekelxx) for other (no water) samples",
    )

    group2.add_argument(
        "-strict_thresh",
        type=float,
        default=50,
        required=False,
        action="store",
        dest="strict_thresh",
        help="Pekel Threshold float (default is 50)",
    )

    group2.add_argument(
        "-display",
        required=False,
        action="store_true",
        help="Display images while running",
    )
    
    group2.add_argument(
        "-save",
        choices=["none", "prim", "aux", "all", "debug"],
        default="none",
        required=False,
        action="store",
        dest="save_mode",
        help="Save all files (debug), only primitives (prim), only pekel and hand (aux), primitives, pekel and hand (all) or only output mask (none)",
    )

    # Samples
    group3.add_argument(
        "-samples_method",
        choices=["smart", "grid", "random"],
        default="smart",
        required=False,
        action="store",
        dest="samples_method",
        help="Select method for choosing learning samples",
    )

    group3.add_argument(
        "-nb_samples_water",
        type=int,
        default=2000,
        required=False,
        action="store",
        dest="nb_samples_water",
        help="Number of samples in water for learning (default is 2000)",
    )

    group3.add_argument(
        "-nb_samples_other",
        type=int,
        default=10000,
        required=False,
        action="store",
        dest="nb_samples_other",
        help="Number of samples in other for learning (default is 10000)",
    )

    group3.add_argument(
        "-nb_samples_auto",
        default=False,
        required=False,
        action="store_true",
        dest="nb_samples_auto",
        help="Auto select number of samples for water and other",
    )

    group3.add_argument(
        "-auto_pct",
        type=float,
        default=0.0002,
        required=False,
        action="store",
        dest="auto_pct",
        help="Percentage of samples points, to use with -nb_samples_auto",
    )

    group3.add_argument(
        "-smart_area_pct",
        type=int,
        default=50,
        required=False,
        action="store",
        dest="smart_area_pct",
        help="For smart method, importance of area for selecting number of samples in each water surface.",
    )

    group3.add_argument(
        "-smart_minimum",
        type=int,
        default=10,
        required=False,
        action="store",
        dest="smart_minimum",
        help="For smart method, minimum number of samples in each water surface.",
    )

    group3.add_argument(
        "-grid_spacing",
        type=int,
        default=40,
        required=False,
        action="store",
        dest="grid_spacing",
        help="For grid method, select samples on a regular grid (40 pixels seems to be a good value)",
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


    # Post processing
    group4.add_argument(
        "-pekel_filter",
        default=True,
        required=False,
        action="store_true",
        dest="pekel_filter",
        help="Postprocess with pekel, only keep surfaces already known by pekel",
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
        "-diameter_closing",
        type=int,
        required=False,
        action="store",
        dest="diameter_closing",
        help="The maximal extension parameter (number of pixels)",
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
    group5.add_argument("file_classif", help="Output classification filename")

    group5.add_argument(
        "-value_classif",
        type=int,
        default=1,
        required=False,
        action="store",
        dest="value_classif",
        help="Output classification value (default is 1)",
    )

    return parser.parse_args()


def main():
    try:
        arguments = getarguments()
        print(arguments)
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
