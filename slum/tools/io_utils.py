#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rasterio as rio
import tracemalloc
import psutil
import linecache
import os


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
    