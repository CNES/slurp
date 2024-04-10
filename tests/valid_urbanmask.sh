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

CMD_MASK="slum_urbanmask"

RES_DIR="/work/CAMPUS/etudes/Masques_CO3D/ValidationTests/Urban/"

DATA_DIR="/work/CAMPUS/etudes/Masques_CO3D/ValidationTests/Images/urban"

# Start
echo "Launch SLUM from `pwd`"

function compute_mask() {
    # Launch watermask computation on $1 with options $2
    image=$1
    image_name=`basename $1`
    shift
    echo "Options : $*"
    # default options
    #options="-remove_false_positive -remove_small_objects 100 -remove_small_holes 50 -binary_closing 3 -save debug"
    options=" -binary_closing 3 -remove_false_positive -remove_small_objects 400 -remove_small_holes 50 -binary_closing 3 -binary_opening 3 -save debug"
    # in order to pass all other options to the script
    ${CMD_MASK} $options $* $image ${RES_DIR}/urbanmask_${image_name}
}

function build_ref() {
    prefix="urbanmask"
    image=`basename $1`
    mask=${RES_DIR}/${prefix}_${image}
    mask_ref=${RES_DIR}/ref_${prefix}_${image}
    cp $mask $mask_ref
}

function check_ref() {
    prefix="urbanmask"
    image=`basename $1`
    mask=${RES_DIR}/${prefix}_${image}
    mask_ref=${RES_DIR}/ref_${prefix}_${image}
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

function help() {
    echo "Watermask unitary tests"
    echo "-h : display this help"
    echo "-r : compute watermask and build ref"
    echo "-f <image> : launch only on <image>"
}

# Get the options
while getopts ":hrf:" option; do
   case $option in
      h) # display Help
          help
          exit;;
      r)
	  build_ref=1
	  echo "Build new references";;
      f)
	  single_test=1
	  image=$OPTARG
	  echo "Launch single test on $OPTARG";;
      \?) # Invalid option
         echo "Error: Invalid option -> use -h to display help"
         exit;;
   esac
done

if [ "$single_test" = "1" ]
then
    echo "Launch unitary test on $image"
    compute_mask $image
    check_ref $image
else
    for im in `ls ${DATA_DIR}/*.tif`;
    do
	compute_mask $im $options
	if [ "$build_ref" = "1" ]
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
