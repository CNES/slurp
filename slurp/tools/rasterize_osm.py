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


"""Module to rasterize image with OTB"""
import argparse
import traceback
import os
import time
import numpy as np
import rasterio
from rasterio import features
import fiona
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import CRS, Transformer
from scipy.ndimage import binary_dilation

import otbApplication as otb

from slurp.tools.constant import COMPRESSION


# def rasterize(args):
#     """Create a rasterized copy of the image passed in arguments"""
#     start_time = time.time()
#
#     app_reproj = otb.Registry.CreateApplication("VectorDataExtractROI")
#     app_reproj.SetParameterString("io.vd", args.osm)
#     app_reproj.SetParameterString("io.in", args.im)
#     app_reproj.SetParameterString("io.out", "tmp_OSM_data.sqlite")
#     app_reproj.ExecuteAndWriteOutput()
#
#     app_raster = otb.Registry.CreateApplication("Rasterization")
#     app_raster.SetParameterString("in", "tmp_OSM_data.sqlite")
#     app_raster.SetParameterString("im", args.im)
#     app_raster.SetParameterString("out", "raster")
#     app_raster.SetParameterString("mode", "binary")
#     app_raster.SetParameterFloat("mode.binary.foreground", 1)
#     app_raster.Execute()
#
#     app_si = otb.Registry.CreateApplication("Superimpose")
#     app_si.SetParameterString("inr", args.im)
#     app_si.SetParameterInputImage(
#         "inm", app_raster.GetParameterOutputImage("out")
#     )
#     if args.dilate > 0:
#         app_si.SetParameterString("out", "superimpose")
#         app_si.Execute()
#         print("Dilatation of vector data / write final result")
#         app_morpho = otb.Registry.CreateApplication(
#             "BinaryMorphologicalOperation"
#         )
#         app_morpho.SetParameterInputImage(
#             "in", app_si.GetParameterOutputImage("out")
#         )
#         app_morpho.SetParameterInt("xradius", args.dilate)
#         app_morpho.SetParameterInt("yradius", args.dilate)
#         app_morpho.SetParameterString(
#             "out",
#             str(
#                 args.out + f"?&gdal:co:TILED=YES&gdal:co:COMPRESS={COMPRESSION}"
#             ),
#         )
#         app_morpho.SetParameterOutputImagePixelType(
#             "out", otb.ImagePixelType_uint8
#         )
#         app_morpho.ExecuteAndWriteOutput()
#     else:
#         print("Write final result")
#         app_si.SetParameterString(
#             "out",
#             str(
#                 args.out + f"?&gdal:co:TILED=YES&gdal:co:COMPRESS={COMPRESSION}"
#             ),
#         )
#         app_si.SetParameterOutputImagePixelType("out", otb.ImagePixelType_uint8)
#         app_si.ExecuteAndWriteOutput()
#
#     os.system("rm tmp_OSM_data.sqlite")
#
#     print("Execution time : " + str(time.time() - start_time))

def rasterize(args):
    """Rasterize OSM vector data onto a reference image, with optional dilation."""
    start_time = time.time()

    # Load reference image
    with rasterio.open(args.im) as src:
        meta = src.meta.copy()
        transform_affine = src.transform
        width = src.width
        height = src.height
        crs = src.crs

    # Load vector features
    with fiona.open(args.osm, 'r') as src_v:
        # Reproject geometries to match image CRS if needed
        if src_v.crs and src_v.crs != crs:
            # Create a transformer from source to target CRS
            transformer = Transformer.from_crs(CRS(src_v.crs), CRS(crs), always_xy=True)
            geometries = [transform(transformer.transform, shape(feat["geometry"])) for feat in src_v]
        else:
            geometries = [shape(feat["geometry"]) for feat in src_v]

    print("Rasterizing vector data...")
    # Rasterize geometries
    rasterized = features.rasterize(
        ((geom, 1) for geom in geometries),
        out_shape=(height, width),
        transform=transform_affine,
        fill=0,
        dtype="uint8"
    )

    if args.dilate > 0:
        print(f"Applying dilation (radius: {args.dilate})...")
        # Dilation kernel size is 2*radius + 1
        structure = np.ones((2 * args.dilate + 1, 2 * args.dilate + 1))
        rasterized = binary_dilation(rasterized, structure=structure).astype(np.uint8)

    print("Saving output raster...")
    meta.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": "uint8",
        "compress": "DEFLATE",
        "tiled": True
    })

    with rasterio.open(args.out, 'w', **meta) as dst:
        dst.write(rasterized, 1)

    print("Execution time:", round(time.time() - start_time, 2), "seconds")


def getarguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Rasterize OSM layer with respect to an input image geographic extent and spacing"
    )

    parser.add_argument(
        "-osm", required=True, action="store", help="OSM building layer"
    )
    parser.add_argument(
        "-im", required=True, action="store", help="Reference image"
    )
    parser.add_argument(
        "-dilate",
        required=False,
        type=int,
        default=0,
        help="Dilatation radius (for line layers - roads, etc.",
    )
    parser.add_argument(
        "-out", required=True, action="store", help="Result file"
    )

    return parser.parse_args()


def main():
    """Main function to rasterize"""
    try:
        arguments = getarguments()
        rasterize(arguments)

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
