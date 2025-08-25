#!/bin/bash

nb_args=3
if [ $# -ne $nb_args ]; then
    echo "Launch SLURP on a 4 bands (R G B NIR) image to compute a vegetation mask"
    echo ""
    echo "Usage : "
    echo "compute_vegetation_mask.sh <MAIN_CONFIG> <PATH_TO_VHR_IMAGE> <VEGETATION_MASK>"
    echo "--"
    echo "Options : "
    echo "export OPT_VEG=''"
    echo "--"
    echo ""
    exit 0
fi

CONF=$1
VHR_IM=$2
VEGETATION_MASK=$3

mkdir out
slurp_prepare $CONF -file_vhr $VHR_IM -mode vegetation $OPT_PREPARE

slurp_vegetationmask out/effective_used_config.json $OPT_VEG -vegetationmask $VEGETATION_MASK

