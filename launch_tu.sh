#!/bin/sh

VHR_IM=$1
DIR_TU=/work/scratch/tanguyy/public/TU_SLUM
cd ${DIR_TU}

mkdir water
mkdir vegetation
mkdir shadow
mkdir urban

slum_watermask ${VHR_IM} water/watermask.tif
slum_vegetationmask -slic True -max_workers 12 ${VHR_IM} vegetation/vegetationmask.tif
slum_shadowmask ${VHR_IM} shadow/shadowmask.tif

slum_urbanmask -use_rgb_layers -building_mask urban/osm_buildings.tif -road_mask urban/osm_roads.tif ${VHR_IM} urban/buildingmask.tif 
slum_urbanmask -use_rgb_layers -building_mask urban/osm_roads.tif -road_mask urban/osm_buildings.tif ${VHR_IM} urban/roadmask.tif 

slum_stackmasks -vegetation vegetation/vegetationmask.tif -water water/watermask.tif -water_pred water/predict.tif -building urban/buildingmask.tif -road urban/roadmask.tif -shadow shadow/shadowmask.tif stack/stack_no_MNH.tif
