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
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz, export_text
import concurrent.futures
from multiprocessing import shared_memory, get_context
from pylab import *
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

def single_bool_profile(input_profiles: list, map_params):
    profile = input_profiles[0]
    profile['count']=1
    profile['dtype']=bool
    profile["compress"] = "lzw"
    
    return profile

def multiple_float_profile(input_profiles: list, map_params):
    profile1 = input_profiles[0]
    profile1['count']=1
    profile1['dtype']=np.float32
    profile1["compress"] = "lzw"
    
    # avoid to modify profile1
    profile2 = deepcopy(profile1)
    profile2['count']=1
    profile2['dtype']=np.float32
    profile2["compress"] = "lzw"
    
    return [profile1, profile2]   

    
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
    print_dataset_infos(ds_filter, desc_filter)
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
    
    #output_scalars= np.append(output_scalars,chunk_output_scalars[0])
    #np.concatenate((output_scalars,chunk_output_scalars[0]))
    output_scalars.append(chunk_output_scalars[0])
   
        
def build_samples(inputBuffer: list, 
                    input_profiles: list, 
                    args: dict) -> list:
    """Build samples."""
    # inputBuffer :[ mask_pekel,valid_stack, mask_hand, im_phr, im_ndvi, im_ndwi]
    # Retrieve number of pixels for each class
    nb_valid_subset = np.count_nonzero(inputBuffer[1])
    nb_water_subset = np.count_nonzero(np.logical_and(inputBuffer[0], inputBuffer[1]))
    nb_other_subset = nb_valid_subset - nb_water_subset
    # Ratio of pixel class compare to the full image ratio
    water_ratio = nb_water_subset/args.nb_valid_water_pixels
    other_ratio = nb_other_subset/args.nb_valid_other_pixels
    # Retrieve number of samples to create for each class in this subset 
    nb_water_subsamples = round(water_ratio*args.nb_samples_water)
    nb_other_subsamples = round(other_ratio*args.nb_samples_other) 
    
    if args.samples_method != "grid":
        # Prepare random water and other samples
        if args.nb_samples_auto:
            nb_water_subsamples = int(nb_water_subset * args.auto_pct)
            nb_other_subsamples = int(nb_other_subset * args.auto_pct)
        
        # Pekel samples
        if args.samples_method == "random":
            rows_pekel, cols_pekel = get_random_indexes_from_masks(
                nb_water_subsamples, inputBuffer[1][0], inputBuffer[0][0]
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
        rows, cols = get_grid_indexes_from_masks(inputBuffer[1], args.grid_spacing)
        if args.save_mode == "debug":
            rc_pekel = [
                (row, col)
                for (row, col) in zip(rows, cols)
                if inputBuffer[0][0][row, col] == 1
            ]
            rc_others = [
                (row, col)
                for (row, col) in zip(rows, cols)
                if inputBuffer[0][0][row, col] != 1
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
    im_stack = np.concatenate((inputBuffer[3],inputBuffer[4],inputBuffer[5],inputBuffer[0]),axis=0)
    samples = np.transpose(im_stack[:, rows, cols])

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
    group1.add_argument("-file_phr", help="PHR filename")

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
        "-filters",
        nargs="+",
        default=[],
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

    group2.add_argument(
        "-simple_ndwi_threshold",
        default = False,
        required = False,
        action = "store_true",
        dest="simple_ndwi_threshold",
        help="Compute water mask as a simple NDWI threshold - useful in arid places where no water is known by Peckel"
    )

    group2.add_argument(
        "-ndwi_threshold",
        default = 0.1,
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

    group3.add_argument(
        "-max_depth",
        type=int,
        default=8,
        required=False,
        action="store",
        dest="max_depth",
        help="Max depth of trees"
    )

    group3.add_argument(
        "-nb_estimators",
        type=int,
        default=100,
        required=False,
        action="store",
        dest="nb_estimators",
        help="Nb of trees in Random Forest"
    )

    group3.add_argument(
        "-n_jobs",
        type=int,
        default=1,
        required=False,
        action="store",
        dest="nb_jobs",
        help="Nb of parallel jobs for Random Forest (1 is recommanded : use n_workers to optimize parallel computing)"
    )
    
    # Post processing
    group4.add_argument(
        "-no_pekel_filter",
        default=False,
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
    group5.add_argument("-file_classif", help="Output classification filename")

    group5.add_argument(
        "-value_classif",
        type=int,
        default=1,
        required=False,
        action="store",
        dest="value_classif",
        help="Output classification value (default is 1)",
    )

    # Parallel computing
    group6.add_argument(
        "-max_mem",
        type=int,
        default=25,
        required=False,
        action="store",
        dest="max_memory",
        help="Max memory permitted for the prediction of the Random Forest (in Gb)"
    )
    
    group6.add_argument(
        "-n_workers",
        type=int,
        default=8,
        required=False,
        action="store",
        dest="nb_workers",
        help="Nb of CPU"
    )

    return parser.parse_args()

 ################ Main function ################
    
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
            
            ### Compute NDVI 
            if args.file_ndvi == None:
                key_ndvi = eoexe.n_images_to_m_images_filter(inputs = [key_phr],
                                                               image_filter = compute_ndvi,
                                                               filter_parameters=args,
                                                               generate_output_profiles = single_float_profile,
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
                key_ndwi = eoexe.n_images_to_m_images_filter(inputs = [key_phr],
                                                               image_filter = compute_ndwi,
                                                               filter_parameters=args,
                                                               generate_output_profiles = single_float_profile,
                                                               stable_margin= 0,
                                                               context_manager = eoscale_manager,
                                                               multiproc_context= "fork",
                                                               filter_desc= "NDWI processing...")         
                if (args.save_mode != "none" and args.save_mode != "aux"):
                    eoscale_manager.write(key = key_ndwi[0], img_path = args.file_classif.replace(".tif","_NDWI.tif"))
            else:
                key_ndwi= [ eoscale_manager.open_raster(raster_path =args.file_ndwi) ]

            
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
            
            time_stack = time.time()
            
            ################ Build samples #################
            
            write = False if (args.save_mode == "none" or args.save_mode == "prim") else True
    
            #### Image Pekel recovery (numpy array, first band)
            if not args.file_pekel:
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
                        write=True,
                    )
                else:
                    args.file_pekel = join(dirname(args.file_classif), "pekel.tif")
                    im_pekel = pekel_recovery(args.file_phr, args.file_pekel, write=True)   
                
                pekel_nodata = 255.0 
                
                
            ds_ref = rio.open(args.file_pekel)
            print_dataset_infos(ds_ref, "PEKEL")
            pekel_nodata = ds_ref.nodata  # contradiction
            key_pekel=eoscale_manager.open_raster(raster_path =args.file_pekel)
            ds_ref.close()
            del ds_ref
        
            args.pekel_nodata=pekel_nodata
                
            ### Pekel valid masks 
            mask_pekel = eoexe.n_images_to_m_images_filter(inputs = [key_pekel] ,
                                                           image_filter = compute_pekel_mask,
                                                           filter_parameters=args,
                                                           generate_output_profiles = multiple_float_profile,
                                                           stable_margin= 0,
                                                           context_manager = eoscale_manager,
                                                           multiproc_context= "fork",
                                                           filter_desc= "Pekel valid mask processing...")

            
            ### Check pekel mask
            select_samples = True
            # - if there are too few values : we threshold NDWI to detect water areas 
            # - if there are even no "supposed water areas" : stop machine learning process (flag select_samples=False)
            local_mask_pekel = eoscale_manager.get_array(mask_pekel[0])
            
            if np.count_nonzero(local_mask_pekel) < 2000:
                print("=> Warning : low water pixel number in Pekel Mask\n")        
                im_ndwi = eoscale_manager.get_array(mask_pekel[0])
                # Threshold NDWI (and then pick-up samples in supposed water areas)
                mask_pekel = compute_mask(im_ndwi, 32767, 1000*args.ndwi_threshold)[0].astype(np.uint8)
                mask_pekel0 = mask_pekel
                if np.count_nonzero(local_mask_pekel) < 2000:
                    print("** WARNING ** too few pixels are considered as water : skip machine learning step")
                    select_samples = False

            ### Image HAND (numpy array, first band)
            if not args.file_hand:
                args.file_hand = join(dirname(args.file_classif), "hand.tif")
                im_hand = hand_recovery(args.file_phr, args.file_hand, write=True)  
                hand_nodata = -9999.0    
                

            ds_hand = rio.open(args.file_hand)
            print_dataset_infos(ds_hand, "HAND")
            hand_nodata = ds_hand.nodata
            key_hand = eoscale_manager.open_raster(raster_path =args.file_hand)
            ds_hand.close()
            del ds_hand    
                
            args.hand_nodata = hand_nodata 
                
            # Create HAND mask 
            mask_hand = eoexe.n_images_to_m_images_filter(inputs = [key_hand],  
                                            image_filter = compute_hand_mask,  #args.hand_strict impossible because of mask_pekel0 not sure
                                            filter_parameters=args,
                                            generate_output_profiles = single_float_profile,
                                            stable_margin= 0,
                                            context_manager = eoscale_manager,
                                            multiproc_context= "fork",
                                            filter_desc= "Hand valid mask processing...")   
            
            
            
            ### Prepare samples   
            if select_samples == False:
                # Not enough supposed water areas : skip sample selection
                # --> we force NDWI threshold
                args.simple_ndwi_threshold = True 
                print("Simple threshold mask NDWI > "+str(args.ndwi_threshold))
                im_predict = compute_mask(eoscale_manager.get_array(key_ndwi[0]), 32767, 1000*args.ndwi_threshold)[0].astype(np.uint8)
                # Force no cleaning with Pekel
                args.no_pekel_filter = True
                time_random_forest = time.time() 
                           
            else:
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
            n_estimators=args.nb_estimators, max_depth=args.max_depth, random_state=0, n_jobs=args.nb_jobs
            )
            print("RandomForest parameters:\n", classifier.get_params(), "\n")
            samples=np.concatenate(samples[:]) # A revoir si possible
            x_samples= samples[:,:-1]
            y_samples= samples[:,-1]
            train_classifier(classifier, x_samples, y_samples)
            print("Dump classifier to model_rf.dump")
            joblib.dump(classifier, "model_rf.dump")
            print_feature_importance(classifier, names_stack)
            gc.collect()
            
     
            
            ######### Predict  ################
            key_predict= eoexe.n_images_to_m_images_filter(inputs = [key_phr, key_ndvi[0], key_ndwi[0],valid_stack_key[0]],       
                                                           image_filter = RF_prediction,
                                                           filter_parameters= {"classifier": classifier},
                                                           generate_output_profiles = single_float_profile,
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
                                                           generate_output_profiles = multiple_float_profile,
                                                           stable_margin= 3,
                                                           context_manager = eoscale_manager,
                                                           multiproc_context= "fork",
                                                           filter_desc= "Post processing...")


            # Save predict and classif image
            final_predict = eoscale_manager.get_array(im_classif[0])[0]
            final_classif = eoscale_manager.get_array(im_classif[1])[0]
            
            save_image(
                final_predict,
                join(dirname(args.file_classif), "predict.tif"),
                args.crs,
                args.transform,
                255,
                args.rpc,
                tags=args.__dict__,
            )

            save_image(
                final_classif,
                args.file_classif,
                args.crs,
                args.transform,
                255,
                args.rpc,
                tags=args.__dict__,
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
