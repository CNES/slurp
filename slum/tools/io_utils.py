#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rasterio as rio


def save_image(
    image, file, crs=None, transform=None, nodata=None, rpc=None, **kwargs
):
    """Save 1 band numpy image to file with lzw compression.
    Note that rio.dtype is string so convert np.dtype to string.
    rpc must be a dictionnary.
    """

    with rio.open(
        file,
        "w",
        driver="GTiff",
        compress="lzw",
        height=image.shape[0],
        width=image.shape[1],
        count=1,
        dtype=str(image.dtype),
        crs=crs,
        transform=transform,
        **kwargs
    ) as dataset:
        dataset.write(image, 1)
        dataset.nodata = nodata

        if rpc:
            dataset.update_tags(**rpc, ns="RPC")

        dataset.close()


def save_image_n_bands(
    image, file, crs=None, transform=None, nodata=None, rpc=None, **kwargs
):
    """Save n bands numpy image to file with lzw compression.
    Note that rio.dtype is string so convert np.dtype to string.
    rpc must be a dictionnary.
    """

    with rio.open(
        file,
        "w",
        driver="GTiff",
        compress="lzw",
        height=image.shape[1],
        width=image.shape[2],
        count=image.shape[0],
        dtype=str(image.dtype),
        crs=crs,
        transform=transform,
        **kwargs
    ) as dataset:
        for i in range(image.shape[0]):
            dataset.write(image[i], i+1)

        dataset.nodata = nodata

        if rpc:
            dataset.update_tags(**rpc, ns="RPC")

        dataset.close()
