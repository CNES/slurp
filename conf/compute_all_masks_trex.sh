#!/bin/bash
#
#SBATCH --job-name=SLUM
#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH --mem=30G # memory pool for all cores
#SBATCH --time=01:30:00
#SBATCH --account=cnes_level2
#SBATCH --export=none

# PHR_IM path to the input image
# OUTPUT_DIR folder containing the outputs
# Compute all the masks for a given image and archive them in a tar
# Example of command to lauch the script :
# sbatch --export="PHR_IM=/work/CAMPUS/etudes/Masques_CO3D/Work/Tests_Celine/Images/Toulouse/xt_PHR_uint16.tif,OUTPUT_DIR=/work/CAMPUS/etudes/Masques_CO3D/Work/Tests_Celine/Tests/Toulouse" compute_all_masks.sh

module load otb/8.1.2-python3.8.4
. /work/scratch/env/raillece/slum_env/bin/activate  # A CHANGER !!

echo ${PHR_IM}

cd $TMPDIR
mkdir -p $OUTPUT_DIR
mkdir -p $TMPDIR/water $TMPDIR/shadows $TMPDIR/urban $TMPDIR/vegetation $TMPDIR/stack $TMPDIR/image

cp ${PHR_IM} ${TMPDIR}/image

filename="$(basename ${PHR_IM})"

# Start
echo "Launch SLUM from `pwd`"

# Watermask
slum_watermask -use_rgb_layers ${TMPDIR}/image/${filename} water/watermask.tif -remove_small_holes 200 -save prim

# Vegetationmask
slum_vegmask_eoscale ${TMPDIR}/image/${filename} vegetation/vegetationmask.tif -ndvi water/ndvi.tif -ndwi water/ndwi.tif -min_ndvi_veg 350 -max_ndvi_noveg 0 -non_veg_clusters -remove_small_objects 100 -binary_dilation 2 -remove_small_holes 100

# Shadowmask
slum_shadowmask ${TMPDIR}/image/${filename} shadows/shadowmask.tif

# Urbanmask
slum_urbanmask -use_rgb_layers ${TMPDIR}/image/${filename} urban/urbanmask.tif -watermask water/watermask.tif -vegetationmask vegetation/vegetationmask.tif -ndvi water/ndvi.tif -ndwi water/ndwi.tif -binary_closing 2 -binary_opening 2 -remove_small_objects 400 -remove_false_positive -save aux

# Stack
slum_stackmasks -vegetation vegetation/vegetationmask.tif -water water/watermask.tif -water_pred water/predict.tif -urban urban/urbanmask_seg.tif -shadow shadows/shadowmask.tif stack/stack_simple.tif

tar cf ${OUTPUT_DIR}/masks.tar water urban vegetation shadows stack

ln -s $PHR_IM ${OUTPUT_DIR}/link_to_VHR_image.tif

# End
