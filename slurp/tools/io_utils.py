#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of SLURP
# (see https://github.com/CNES/slurp).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Brings together useful functions common to the different scripts"""
import json

import matplotlib.pyplot as plt
import numpy as np
import logging
import rasterio as rio
from slurp.tools.pydantic_class import load_config, MainConfig, UserConfig

from slurp.tools.constant import COMPRESSION, DRIVER

logger = logging.getLogger("slurp")

def read_json(
    main_config_file: str, keys: list, user_config_file: str = None
) -> dict:
    """
    Read JSON config files

    :param str main_config_file: Path to the main JSON config file
    :param list keys: Keys to read in the JSON files
    :param str user_config_file: Path to the overload JSON config file (None by default)
    :returns: dictionary of arguments
    """
    # Read the JSON data from the main config
    try:
        config = load_config(main_config_file, MainConfig)
        full_args = config.dict()
        argsdict = full_args[keys[0]]
        for key in keys[1:]:
            argsdict.update(full_args[key])

    except FileNotFoundError:
        logger.error(f"File {main_config_file} not found.")
    except json.JSONDecodeError:
        logger.error(
            f"Error decoding JSON data from {main_config_file}. Please check the file format."
        )

    if user_config_file:
        # Read the JSON data from the input file
        try:
            config = load_config(user_config_file, UserConfig)
            full_args = config.dict()
            for k in full_args.keys():
                argsdict.update(full_args[k])

        except FileNotFoundError:
            logger.error(f"File {user_config_file} not found.")
        except json.JSONDecodeError:
            logger.error(
                f"Error decoding JSON data from {user_config_file}. Please check the file format."
            )

    return argsdict


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
    """
    Save 1 band numpy image to file with deflate compression.
    Note that rio.dtype is string so convert np.dtype to string.
    rpc must be a dictionary.
    """

    dataset = rio.open(
        file,
        "w",
        driver=DRIVER,
        compress=COMPRESSION.lower(),
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


def save_image_n_bands(
    image, file, crs=None, transform=None, nodata=None, rpc=None, **kwargs
):
    """
    Save n bands numpy image to file with lzw compression.
    Note that rio.dtype is string so convert np.dtype to string.
    rpc must be a dictionary.
    """

    with rio.open(
        file,
        "w",
        driver=DRIVER,
        compress=COMPRESSION.lower(),
        height=image.shape[1],
        width=image.shape[2],
        count=image.shape[0],
        dtype=str(image.dtype),
        crs=crs,
        transform=transform,
        **kwargs,
    ) as dataset:
        for i in range(image.shape[0]):
            dataset.write(image[i], i + 1)

        dataset.nodata = nodata

        if rpc:
            dataset.update_tags(**rpc, ns="RPC")

        dataset.close()


def show_images(image1, title1, image2, title2, **kwargs):
    """Show 2 images with matplotlib."""

    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(14, 7), sharex="all", sharey="all"
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

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7), sharey="all")

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

    axe.plot(np.arange(-1000, 1001, step=10), hist1, label=title1, **kwargs)
    axe.plot(np.arange(-1000, 1001, step=10), hist2, label=title2, **kwargs)
    axe.plot(np.arange(-1000, 1001, step=10), hist3, label=title3, **kwargs)
    axe.plot(np.arange(-1000, 1001, step=10), hist4, label=title4, **kwargs)

    fig.tight_layout()
    plt.legend()
    plt.show()
