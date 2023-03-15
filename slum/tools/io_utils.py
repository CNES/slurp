#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rasterio as rio
import tracemalloc
import psutil
import linecache
import os

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

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def display_mem(step):
    mem_used = psutil.Process().memory_info().rss / (1024 * 1024)
    print(">>>"+str(step)+"\t >>> Mem used : \t"+str(mem_used)+" Mb")
