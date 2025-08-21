#!/bin/bash

nb_args=3
if [ $# -ne $nb_args ]; then
    echo "Launch SLURP on a 4 bands (R G B NIR) image and get simple land use mask"
    echo "External data is retrieved and superimposed with OTB"
    echo ""
    echo "Usage : "
    echo "compute_slurp_masks.sh <MAIN_CONFIG> <PATH_TO_VHR_IMAGE> <LAND_USE_MASK>"
    echo "--"
    echo "Options : "
    echo "export OPT_PREPARE='sensor_mode=true'"
    echo "(idem with OPT_WATER, OPT_VEG, OPT_SHADOW, OPT_URBAN, OPT_STACK"
    echo "--"
    echo ""
    echo "OTB is supposed to be properly installed in your environment"
    echo ""
    exit 0
fi

CONF=$1
VHR_IM=$2
LAND_USE_MASK=$3

echo ${VHR_IM}

mkdir out

# Start
echo "Launch SLURP from `pwd`"

# Superimpose Pekel, Hand and WSF with OTB
# 
# /!\ Adapt path depending on where your global Pekel database (resp. HAND, WSF) is located
otbcli_Superimpose -inr ${VHR_IM} -inm /work/datalake/static_aux/MASQUES/PEKEL/data2021/occurrence/occurrence.vrt -out "out/pekel.tif?&gdal:co:TILED=YES&gdal:co:COMPRESS=DEFLATE" uint8 -interpolator nn
otbcli_Superimpose -inr ${VHR_IM} -inm /work/datalake/static_aux/MASQUES/HAND_MERIT/hnd.vrt -out "out/hand.tif?&gdal:co:TILED=YES&gdal:co:COMPRESS=DEFLATE" 
otbcli_Superimpose -inr ${VHR_IM} -inm /work/datalake/static_aux/MASQUES/WSF/WSF2019_v1/WSF2019_v1.vrt -out "out/wsf.tif?&gdal:co:TILED=YES&gdal:co:COMPRESS=DEFLATE" uint8 -interpolator nn

# Prepare
slurp_prepare ${CONF} -file_vhr ${VHR_IM} ${OPT_PREPARE}

# Watermask
slurp_watermask out/effective_used_config.json ${OPT_WATER}

# Vegetationmask
slurp_vegetationmask out/effective_used_config.json ${OPT_VEG}

# Shadowmask
slurp_shadowmask out/effective_used_config.json ${OPT_SHADOW}

# Urbanmask (without post-processing)
slurp_urbanmask out/effective_used_config.json ${OPT_URBAN}

# Stack
slurp_stackmasks out/effective_used_config.json ${OPT_STACK} -stackmask ${LAND_USE_MASK}
