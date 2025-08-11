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
import os
import json
import time
import traceback
import pathlib
from os import makedirs, path
import logging

import otbApplication as otb

from slurp.tools.constant import COMPRESSION
from slurp.tools import utils

logger = logging.getLogger("slurp")

def rasterize(osm : str, im : str, dilate : int = 0, out : str = "raster"):
    """Create a rasterized copy of the image passed in arguments"""
    start_time = time.time()

    app_reproj = otb.Registry.CreateApplication("VectorDataExtractROI")
    app_reproj.SetParameterString("io.vd", osm)
    app_reproj.SetParameterString("io.in", im)
    app_reproj.SetParameterString("io.out", "tmp_OSM_data.sqlite")
    app_reproj.ExecuteAndWriteOutput()

    app_raster = otb.Registry.CreateApplication("Rasterization")
    app_raster.SetParameterString("in", "tmp_OSM_data.sqlite")
    app_raster.SetParameterString("im", im)
    app_raster.SetParameterString("out", "raster")
    app_raster.SetParameterString("mode", "binary")
    app_raster.SetParameterFloat("mode.binary.foreground", 1)
    app_raster.Execute()

    app_si = otb.Registry.CreateApplication("Superimpose")
    app_si.SetParameterString("inr", im)
    app_si.SetParameterInputImage(
        "inm", app_raster.GetParameterOutputImage("out")
    )
    if dilate > 0:
        app_si.SetParameterString("out", "superimpose")
        app_si.Execute()
        logger.info("Dilatation of vector data / write final result")
        app_morpho = otb.Registry.CreateApplication(
            "BinaryMorphologicalOperation"
        )
        app_morpho.SetParameterInputImage(
            "in", app_si.GetParameterOutputImage("out")
        )
        app_morpho.SetParameterInt("xradius", dilate)
        app_morpho.SetParameterInt("yradius", dilate)
        app_morpho.SetParameterString(
            "out",
            str(
                out + f"?&gdal:co:TILED=YES&gdal:co:COMPRESS={COMPRESSION}"
            ),
        )
        app_morpho.SetParameterOutputImagePixelType(
            "out", otb.ImagePixelType_uint8
        )
        app_morpho.ExecuteAndWriteOutput()
    else:
        logger.info("Write final result")
        app_si.SetParameterString(
            "out",
            str(
                out + f"?&gdal:co:TILED=YES&gdal:co:COMPRESS={COMPRESSION}"
            ),
        )
        app_si.SetParameterOutputImagePixelType("out", otb.ImagePixelType_uint8)
        app_si.ExecuteAndWriteOutput()

    os.system("rm tmp_OSM_data.sqlite")

    logger.info("Execution time : " + str(time.time() - start_time))


def getarguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Rasterize OSM layer with respect to an input image geographic extent and spacing"
    )

    parser.add_argument(
        "-osm", required=True, action="store", help="OSM building layer"
    )
    parser.add_argument("-log_f",
                        "--logs_to_file",
                        action="store_true",
                        help="Store all logs to a file, instead of stdout",
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
    args = parser.parse_args()

    arglist = []
    for arg in parser._actions:
        if arg.dest not in ["help"]:
            arglist.append(arg.dest)

    with open("args_list.json", 'w') as f:
        json.dump(arglist, f)

    return vars(args)


def rasterize_osm(osm : str, logs_to_file: bool, im : str, dilate : int = 0, out : str = "raster"):
    """Main function to rasterize"""
    try:
        if logs_to_file:
            config_file = pathlib.Path("slurp/tools/logs/out2json.json")
            if not path.exists("logs"):
                makedirs("logs")
        else:
            config_file = pathlib.Path("slurp/tools/logs/out2stdout.json")
        utils.setup_logging(config_file)

        rasterize(osm, im, dilate, out)

    except FileNotFoundError as fnfe_exception:
        logger.error("FileNotFoundError", fnfe_exception)

    except PermissionError as pe_exception:
        logger.error("PermissionError", pe_exception)

    except ArithmeticError as ae_exception:
        logger.error("ArithmeticError", ae_exception)

    except MemoryError as me_exception:
        logger.error("MemoryError", me_exception)

    except Exception as exception:  # pylint: disable=broad-except
        logger.error("oups...", exception)
        traceback.print_exc()


def main():
    """
    Main function to run the osm rasterization.
    It parses the command line arguments and calls the rasterize_osm function.
    """
    args = getarguments()
    rasterize_osm(**args)

if __name__ == "__main__":
    main()
