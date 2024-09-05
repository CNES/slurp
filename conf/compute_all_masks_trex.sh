#!/bin/bash
#
#SBATCH --job-name=SLURP
#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH --mem=30G # memory pool for all cores
#SBATCH --time=01:30:00
#SBATCH --account=cnes_level2

# PHR_IM path to the input image
# OUTPUT_DIR folder containing the outputs
# Compute all the masks for a given image and archive them in a tar
# Example of command to lauch the script :
# sbatch --export="PHR_IM=path_to_VHR_uint16_image.tif,OUTPUT_DIR=path_to_outputmasks_directory,OPT_PREPARE='',OPT_VEG='',OPT_STACK=''" compute_all_masks.sh


. ~/bin/init_slurp.sh
module load monitoring/1.0
start_monitoring.sh --name SLURP_all_masks

echo ${PHR_IM}

cd $TMPDIR
mkdir -p $OUTPUT_DIR
mkdir -p $TMPDIR/out

cp ${PHR_IM} ${TMPDIR}

filename="$(basename ${PHR_IM})"

main_config="/home/qt/tanguyy/SRC/slurp/conf/main_config.json"

# Start
echo "Launch SLURP from `pwd`"

# Prepare
slurp_prepare ${main_config} -file_vhr ${TMPDIR}/${filename} ${OPT_PREPARE}
# Watermask
slurp_watermask out/used_config.json ${OPT_WATER}

# Vegetationmask
slurp_vegetationmask out/used_config.json ${OPT_VEG}

# Shadowmask
slurp_shadowmask out/used_config.json

# Urbanmask (without post-processing)
slurp_urbanmask out/used_config.json ${OPT_URBAN}

# Stack
slurp_stackmasks out/used_config.json ${OPT_STACK}

stop_monitoring.sh --name SLURP_all_masks

current_date=`date +%F`

tar cf ${OUTPUT_DIR}/masks_${current_date}.tar out

ln -s $PHR_IM ${OUTPUT_DIR}/${filename}

sed "s,PATH_TO_TAR,${OUTPUT_DIR}/masks_${current_date}.tar," ~/SRC/slurp/conf/template_project.qgs | sed "s,LINK_TO_THR,${OUTPUT_DIR}/${TMPDIR}/${filename}," > ${OUTPUT_DIR}/my_project.qgs

echo "QGIS project available : check the geographical extent (Apply image CRS to other layers), check the image THR layer (fix percentiles to 2/98) and enjoy !"
# End
