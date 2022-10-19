#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import dirname, join
from subprocess import call

import traceback
import argparse
import otbApplication as otb
import time

def rasterize(args):
    start_time = time.time()
    appReproj = otb.Registry.CreateApplication('VectorDataReprojection')
    appReproj.SetParameterString("in.vd",args.osm)
    appReproj.SetParameterString("out.proj.image.in",args.im)
    appReproj.SetParameterString("out.vd","tmp_OSM_data.sqlite")
    appReproj.ExecuteAndWriteOutput()

    appRaster = otb.Registry.CreateApplication('Rasterization')
    appRaster.SetParameterString("in", "tmp_OSM_data.sqlite")
    appRaster.SetParameterString("im", args.im)
    appRaster.SetParameterString("out", "raster")
    appRaster.SetParameterString("mode", "binary")
    appRaster.SetParameterFloat("mode.binary.foreground",1)
    appRaster.Execute()

    appSI = otb.Registry.CreateApplication("Superimpose")
    appSI.SetParameterString("inr",args.im)
    appSI.SetParameterInputImage("inm",appRaster.GetParameterOutputImage("out"))
    if args.dilate > 0:
        appSI.SetParameterString("out", "superimpose")
        appSI.Execute()
        print("Dilatation of vector data / write final result")
        appMorpho = otb.Registry.CreateApplication("BinaryMorphologicalOperation")
        appMorpho.SetParameterInputImage("in",appSI.GetParameterOutputImage("out"))
        appMorpho.SetParameterInt("xradius",args.dilate)
        appMorpho.SetParameterInt("yradius",args.dilate)
        appMorpho.SetParameterString("out", str(args.out+"?&gdal:co:TILED=YES&gdal:co:COMPRESS=DEFLATE"))
        appMorpho.SetParameterOutputImagePixelType("out",otb.ImagePixelType_uint8)
        appMorpho.ExecuteAndWriteOutput()
    else:
        print("Write final result")
        appSI.SetParameterString("out", str(args.out+"?&gdal:co:TILED=YES&gdal:co:COMPRESS=DEFLATE"))
        appSI.SetParameterOutputImagePixelType("out",otb.ImagePixelType_uint8)
        appSI.ExecuteAndWriteOutput()

    os.system("rm tmp_OSM_data.sqlite")

    print("Execution time : "+str(time.time() - start_time))



def getarguments():
    """ Parse command line arguments. """

    parser = argparse.ArgumentParser(description='Rasterize OSM layer with respect to an input image geographic extent and spacing')

    parser.add_argument('-osm', required=True, action='store', dest='osm',
                        help='OSM building layer')
    parser.add_argument('-im', required=True, action='store', dest='im',
                       help='Reference image')
    parser.add_argument('-dilate',required=False,type=int, default=0,
                        help='Dilatation radius (for line layers - roads, etc.')
    parser.add_argument('-out', required=True, action='store', dest='out',
                        help='Result file')

    return parser.parse_args()


def main():
    try:
        arguments = getarguments()
        rasterize(arguments)

    except FileNotFoundError as fnfe_exception:
        print('FileNotFoundError', fnfe_exception)

    except PermissionError as pe_exception:
        print('PermissionError', pe_exception)

    except ArithmeticError as ae_exception:
        print('ArithmeticError', ae_exception)

    except MemoryError as me_exception:
        print('MemoryError', me_exception)

    except Exception as exception: # pylint: disable=broad-except
        print('oups...', exception)
        traceback.print_exc()

if __name__ == '__main__':
    main()
