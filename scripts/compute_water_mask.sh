#!/bin/bash

nb_args=3
if [ $# -ne $nb_args ]; then
    echo "Launch SLURP on a 4 bands (R G B NIR) image to compute a water mask"
    echo "External data is retrieved and superimposed with OTB"
    echo ""
    echo "Usage : "
    echo "compute_water_mask.sh <MAIN_CONFIG> <PATH_TO_VHR_IMAGE> <WATER_MASK>"
    echo "--"
    echo "Options : "
    echo "export OPT_PREPARE=''"
    echo "export OPT_WATER=''"
    echo "--"
    echo ""
    echo "OTB is supposed to be properly installed in your environment"
    echo ""
    exit 0
fi

CONF=$1
VHR_IM=$2
WATER_MASK=$3

mkdir out

# Superimpose Pekel, Hand and WSF with OTB
# 
# /!\ Adapt path depending on where your global Pekel database (resp. HAND) is located
otbcli_Superimpose -inr ${VHR_IM} -inm /work/datalake/static_aux/MASQUES/PEKEL/data2021/occurrence/occurrence.vrt -out "out/pekel.tif?&gdal:co:TILED=YES&gdal:co:COMPRESS=DEFLATE" uint8 -interpolator nn
otbcli_Superimpose -inr ${VHR_IM} -inm /work/datalake/static_aux/MASQUES/HAND_MERIT/hnd.vrt -out "out/hand.tif?&gdal:co:TILED=YES&gdal:co:COMPRESS=DEFLATE" 
slurp_prepare $CONF -file_vhr $VHR_IM -mode water $OPT_PREPARE

slurp_watermask out/effective_used_config.json $OPT_WATER -watermask $WATER_MASK

