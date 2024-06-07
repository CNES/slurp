#!/bin/bash
#
#SBATCH --job-name=SLURP
#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH --mem=30G # memory pool for all cores
#SBATCH --time=01:00:00
#SBATCH --account=cnes_level2
#SBATCH --export=none

# MASK_TAR path to the tar folder
# OSM path to the OSM file
# OUTPUT_DIR folder containing the outputs
# Compute scores for a urbanmask_seg.tif file inside a tar
# Example of command to lauch the script :
# sbatch --export="MASK_TAR=/work/CAMPUS/etudes/Masques_CO3D/Work/Tests_Celine/Tests/Toulouse/masks.tar,OSM=/work/CAMPUS/etudes/Masques_CO3D/Work/Tests_Celine/OSM/toulouse.tif,OUTPUT_DIR=/work/CAMPUS/etudes/Masques_CO3D/Work/Tests_Celine/Tests/Toulouse" scores_from_tar_trex.sh

module load otb/8.1-python3.8.4
. /softs/projets/pluto/slurp_env_v2/bin/activate

echo "SCORES SLURP"
echo ${MASK_TAR}

cd $TMPDIR
mkdir -p $OUTPUT_DIR
tar -C $TMPDIR/ -xvf $MASK_TAR urban

# Start
echo "Launch SLURP from `pwd`"

slurp_scores -im $TMPDIR/urban/urbanmask_seg.tif -gt $OSM -out compare_OSM_pred.tif

cp compare_OSM_pred.tif ${OUTPUT_DIR}/

# End
