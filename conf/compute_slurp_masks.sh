#!/bin/bash
#
#SBATCH --job-name=SLURP
#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH --mem=60G # memory pool for all cores
#SBATCH --time=01:30:00
#SBATCH --account=cnes_level2

# PHR_IM path to the input image
# OUTPUT_DIR folder containing the outputs
# Compute all the masks for a given image and archive them in a tar
# Example of command to lauch the script :
# sbatch --export="OPT_PREPARE='',OPT_VEG='',OPT_STACK=''" /softs/projets/pluto/slurp/compute_slurp_masks.sh <PATH TO YOUR IMAGE> <OUTPUT DIR>

nb_args=2
if [ $# -ne $nb_args ]; then
    echo "Launch SLURP on a 4 bands (R G B NIR) image and get simple land use mask"
    echo ""
    echo "Usage : "
    echo "compute_slurp_masks.sh <PATH_TO_VHR_IMAGE> <OUTPUT_DIR>"
    echo "--"
    echo "Options : "
    echo "export OPT_PREPARE='sensor_mode=true'"
    echo "(idem with OPT_WATER, OPT_VEG, OPT_SHADOW, OPT_URBAN, OPT_STACK"
    echo "--"
    echo "Launch on a cluster node (slurm) :"
    echo "sbatch --export='OPT_PREPARE=specific_options' /softs/projets/pluto/slurp/compute_slurp_masks.sh <PATH_TO_VHR_IMAGE> <OUTPUT_DIR>"
    echo ""
    exit 0
fi

# source your own environment to use a personal SLURP installation
. /softs/projets/pluto/slurp/init_slurp.sh
#. /home/qt/tanguyy/bin/init_slurp.sh
module load monitoring
start_monitoring.sh --name SLURP_all_masks

VHR_IM=$1
OUTPUT_DIR=$2

echo ${VHR_IM}

cd $TMPDIR

mkdir -p $OUTPUT_DIR
mkdir -p $TMPDIR/out

rm -rf $TMPDIR/out/*

cp ${VHR_IM} ${TMPDIR}

filename="$(basename ${VHR_IM})"

main_config="/softs/projets/pluto/slurp/main_config.json"
#main_config="/home/qt/tanguyy/SRC/slurp/conf/main_config.json"

# Start
echo "Launch SLURP from `pwd`"

# Superimpose Pekel, Hand and WSF with OTB
otbcli_Superimpose -inr ${VHR_IM} -inm /work/datalake/static_aux/MASQUES/PEKEL/data2021/occurrence/occurrence.vrt -out "out/pekel.tif?&gdal:co:TILED=YES&gdal:co:COMPRESS=DEFLATE" uint8
otbcli_Superimpose -inr ${VHR_IM} -inm /work/datalake/static_aux/MASQUES/HAND_MERIT/hnd.vrt -out "out/hand.tif?&gdal:co:TILED=YES&gdal:co:COMPRESS=DEFLATE" 
otbcli_Superimpose -inr ${VHR_IM} -inm /work/datalake/static_aux/MASQUES/WSF/WSF2019_v1/WSF2019_v1.vrt -out "out/wsf.tif?&gdal:co:TILED=YES&gdal:co:COMPRESS=DEFLATE" uint8

# Prepare
slurp_prepare ${main_config} -file_vhr ${TMPDIR}/${filename} ${OPT_PREPARE}

exit
# Watermask
slurp_watermask out/effective_used_config.json ${OPT_WATER}

# Vegetationmask
slurp_vegetationmask out/effective_used_config.json ${OPT_VEG}

# Shadowmask
slurp_shadowmask out/effective_used_config.json ${OPT_SHADOW}

# Urbanmask (without post-processing)
slurp_urbanmask out/effective_used_config.json ${OPT_URBAN}

# Stack
slurp_stackmasks out/effective_used_config.json ${OPT_STACK}

current_date=`date +%F`
stop_monitoring.sh --name SLURP_all_masks

tar cf ${OUTPUT_DIR}/masks_${current_date}.tar out

ln -s $VHR_IM ${OUTPUT_DIR}/${filename}

sed "s,PATH_TO_TAR,${OUTPUT_DIR}/masks_${current_date}.tar," /softs/projets/pluto/slurp/template_project.qgs | sed "s,LINK_TO_THR,${OUTPUT_DIR}/${filename}," > ${OUTPUT_DIR}/slurp_masks_${current_date}.qgs

echo "QGIS project available : check the geographical extent (Apply image CRS to other layers), check the image THR layer (fix percentiles to 2/98) and enjoy !"
# End
