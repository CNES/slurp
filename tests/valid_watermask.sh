#!/bin/bash
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH --mem=30G # memory pool for all cores
#SBATCH --time=01:30:00
#SBATCH --account=cnes_level2

# PHR_IM path to the input image
# OUTPUT_DIR folder containing the outputs
# Compute all the masks for a given image and archive them in a tar
# Example of command to lauch the script :
# sbatch valid_watermask.sh

module load otb/9.0.0rc2-python3.8
. /work/scratch/env/tanguyy/venv/slum_otb9/bin/activate

CMD_MASK="python /home/qt/tanguyy/SRC/slum/slum/water/watermask_eoscale.py"
RES_DIR="/work/CAMPUS/etudes/Masques_CO3D/ValidationTests/Water/"

DATA_DIR="/work/CAMPUS/etudes/Masques_CO3D/ValidationTests/Images"

# Start
echo "Launch SLUM from `pwd`"

# Watermask
# TODO : remove use_rgb_layers (always true)

#${CMD_MASK} -use_rgb_layers ${DATA_DIR}/xt_no_peckel_but_water.tif ${RES_DIR}/watermask_no_peckel_area/watermask.tif -ndwi_threshold -0.1
#${CMD_MASK} -use_rgb_layers ${DATA_DIR}/xt_no_peckel_no_water.tif ${RES_DIR}/watermask_dry_area/watermask.tif -save debug

function compute_mask() {
    # Launch watermask computation on $1 with options $2
    image=$1
    image_name=`basename $1`
    shift
    echo "Options : $*"
    # in order to pass all other options to the script
    ${CMD_MASK} $* $image ${RES_DIR}/watermask_${image_name}
}

function build_ref() {
    image=`basename $1`
    mask=${RES_DIR}/watermask_${image}
    mask_ref=${RES_DIR}/watermask_ref_${image}
    if [ -f "$mask_ref" ]
    then
	echo "$mask_ref already exists"
    else
	echo "create new ref $mask_ref"
	cp $mask $mask_ref
    fi
}

function check_ref() {
    image=`basename $1`
    mask=${RES_DIR}/watermask_${image}
    mask_ref=${RES_DIR}/watermask_ref_${image}
    if [ -f $mask_ref ] && [ -f $mask ]
    then
	md5_res=`md5sum $mask`
	md5_ref=`md5sum $mask_ref`
	if [ "${md5_res:0:32}" = "${md5_ref:0:32}" ]
	then
	    echo "OK - $mask"
	else
	    echo "!!! NOK $mask differs from $mask_ref - please check !!"	    
	fi
    else
	echo "Warning : $mask_ref does not exist"
    fi
}

if [ -f "$1" ]
then
    echo "Launch unitary test on $1"
    im=$1
    shift
    options=$*
    compute_mask $im $options
    check_ref $im
else
    for im in `ls ${DATA_DIR}/*.tif`;
    do
	options="-use_rgb_layers"
	compute_mask $im $options
	if [ $1="-build_ref" ]
	then
	    build_ref $im
	fi
    done

    for im in `ls ${DATA_DIR}/*.tif`;
    do
	check_ref $im
    done
fi	  
	  



# End
