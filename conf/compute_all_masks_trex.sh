#!/bin/bash
#
#SBATCH --job-name=SLUM
#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH --mem=30G # memory pool for all cores
#SBATCH --time=01:30:00
#SBATCH --account=cnes_level2

# PHR_IM path to the input image
# OUTPUT_DIR folder containing the outputs
# Compute all the masks for a given image and archive them in a tar
# Example of command to lauch the script :
# sbatch --export="PHR_IM=path_to_VHR_uint16_image.tif,OUTPUT_DIR=path_to_outputmasks_directory,CLUSTERS_VEG=4,CLUSTERS_LOW_VEG=2" compute_all_masks.sh

. /home/qt/tanguyy/bin/init_slum.sh
#module load otb/8.1.2-python3.8.4
#. /work/scratch/env/tanguyy/venv/slum_vre/bin/activate

module load monitoring/1.0
start_monitoring.sh --name SLUM_all_masks

echo ${PHR_IM}

cd $TMPDIR
mkdir -p $OUTPUT_DIR
mkdir -p $TMPDIR/water $TMPDIR/shadows $TMPDIR/urban $TMPDIR/vegetation $TMPDIR/stack $TMPDIR/image

cp ${PHR_IM} ${TMPDIR}/image

filename="$(basename ${PHR_IM})"

# Start
echo "Launch SLUM from `pwd`"

# Watermask
slum_watermask -remove_small_holes 100 -binary_closing 2 -save prim ${TMPDIR}/image/${filename} water/watermask.tif 

# Vegetationmask
slum_vegetationmask -ndvi water/watermask_NDVI.tif -ndwi water/watermask_NDWI.tif -non_veg_clusters -remove_small_objects 100 -binary_dilation 2 -remove_small_holes 100 -nbclusters ${CLUSTERS_VEG} -nbclusters_low ${CLUSTERS_LOW_VEG} ${TMPDIR}/image/${filename} vegetation/vegetationmask.tif 

# Shadowmask
slum_shadowmask ${TMPDIR}/image/${filename} shadows/shadowmask.tif -binary_opening 2 -remove_small_objects 50

# Urbanmask
slum_urbanmask  -watermask water/watermask.tif -vegetationmask vegetation/vegetationmask.tif  -ndvi water/watermask_NDVI.tif -ndwi water/watermask_NDWI.tif -binary_closing 2 -binary_opening 2 -remove_small_objects 100 -remove_small_holes 100 -remove_false_positive -confidence_threshold 70 -shadowmask shadows/shadowmask.tif -save debug  ${TMPDIR}/image/${filename} urban/urbanmask.tif

# Stack
slum_stackmasks -vegetation vegetation/vegetationmask.tif -water water/watermask.tif -water_pred water/predict.tif -urban urban/urbanmask.tif -shadow shadows/shadowmask.tif stack/stack_simple.tif

stop_monitoring.sh --name SLUM_all_masks

tar cf ${OUTPUT_DIR}/masks_${CLUSTERS_VEG}_${CLUSTERS_LOW_VEG}.tar water urban vegetation shadows stack

ln -s $PHR_IM ${OUTPUT_DIR}/link_to_VHR_image.tif

# sed "s,PATH_TO_TAR,${OUTPUT_DIR}/masks_${CLUSTERS_VEG}_${CLUSTERS_LOW_VEG}.tar," ~/SRC/slum/conf/template_project.qgs | sed "s,vegetationmask.tif,vegetationmask_${CLUSTERS_VEG}_${CLUSTERS_LOW_VEG}.tif," | sed "s,LINK_TO_THR,${OUTPUT_DIR}/link_to_VHR_image.tif," > ${OUTPUT_DIR}/my_project.qgs
sed "s,PATH_TO_TAR,${OUTPUT_DIR}/masks_${CLUSTERS_VEG}_${CLUSTERS_LOW_VEG}.tar," ~/SRC/slum/conf/template_project.qgs | sed "s,LINK_TO_THR,${OUTPUT_DIR}/link_to_VHR_image.tif," > ${OUTPUT_DIR}/my_project.qgs

echo "QGIS project available : check the geographical extent (Apply image CRS to other layers), check the image THR layer (fix percentiles to 2/98) and enjoy !"
# End
