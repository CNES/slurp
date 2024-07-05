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
# sbatch --export="PHR_IM=path_to_VHR_uint16_image.tif,OUTPUT_DIR=path_to_outputmasks_directory,CLUSTERS_VEG=4,CLUSTERS_LOW_VEG=2" compute_all_masks.sh


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
slurp_prepare ${main_config} -file_vhr ${TMPDIR}/${filename}
# Watermask
slurp_watermask ${main_config} -file_vhr ${TMPDIR}/${filename} 

# Vegetationmask
slurp_vegetationmask ${main_config} -file_vhr ${TMPDIR}/${filename}


# Shadowmask
# slurp_shadowmask ${TMPDIR}/image/${filename} shadows/shadowmask.tif -binary_opening 2 -remove_small_objects 100 -th_rgb 0.2 -th_nir 0.2 -watermask water/watermask.tif 
slurp_shadowmask ${main_config} -file_vhr ${TMPDIR}/${filename}

# Urbanmask (without post-processing)
# slurp_urbanmask  -watermask water/watermask.tif -vegetationmask vegetation/vegetationmask.tif  -ndvi water/watermask_NDVI.tif -ndwi water/watermask_NDWI.tif -binary_closing 5 -binary_opening 2 -shadowmask shadows/shadowmask.tif ${TMPDIR}/image/${filename} urban/urbanmask.tif -nb_samples_urban 10000 -nb_samples_other 10000 -binary_dilation 5 
slurp_urbanmask ${main_config} -file_vhr ${TMPDIR}/${filename}

# Stack
# slurp_stackmasks ${TMPDIR}/image/${filename} stack/stack_simple.tif -vegmask vegetation/vegetationmask.tif -watermask water/watermask.tif -waterpred water/watermask.tif -urban_proba urban/urbanmask_proba.tif  -shadow shadows/shadowmask.tif -wsf urban/wsf.tif -remove_small_objects 300 -binary_closing 3 -binary_opening 3 -remove_small_holes 300 -building_erosion 2 -bonus_gt 10 -malus_shadow 10

slurp_stackmasks ${main_config} -file_vhr ${TMPDIR}/${filename} -waterpred toto.tif 

stop_monitoring.sh --name SLURP_all_masks

current_date=`date +%F`

tar cf ${OUTPUT_DIR}/masks_${CLUSTERS_VEG}_${CLUSTERS_LOW_VEG}_${current_date}.tar out

ln -s $PHR_IM ${OUTPUT_DIR}/${TMPDIR}/${filename}

# sed "s,PATH_TO_TAR,${OUTPUT_DIR}/masks_${CLUSTERS_VEG}_${CLUSTERS_LOW_VEG}_${current_date}.tar," ~/SRC/slurp/conf/template_project.qgs | sed "s,vegetationmask.tif,vegetationmask_${CLUSTERS_VEG}_${CLUSTERS_LOW_VEG}.tif," | sed "s,LINK_TO_THR,${OUTPUT_DIR}/${TMPDIR}/${filename}," > ${OUTPUT_DIR}/my_project.qgs
sed "s,PATH_TO_TAR,${OUTPUT_DIR}/masks_${CLUSTERS_VEG}_${CLUSTERS_LOW_VEG}_${current_date}.tar," ~/SRC/slurp/conf/template_project.qgs | sed "s,LINK_TO_THR,${OUTPUT_DIR}/${TMPDIR}/${filename}," > ${OUTPUT_DIR}/my_project.qgs

echo "QGIS project available : check the geographical extent (Apply image CRS to other layers), check the image THR layer (fix percentiles to 2/98) and enjoy !"
# End
