<div align="center">
  <a href="https://gitlab.cnes.fr/pluto/slum"><img src="docs/source/images/logo_SLUM_256.png" alt="SLUM" title="SLUM"  width="20%"></a>

<h4>slum</h4>

[![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)


<p>
  <a href="#overview">Overview</a> •
  <a href="#features">Features</a> •
  <a href="#install">Install</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#contribution">Contribution</a> •
  <a href="#references">References</a>
</p>
</div>

## Overview

**SLUM** : **S**mart **L**and **U**se **M**asks

SLUM proposes different algorithms to perform Land Use/Land Cover masks, with few data. Several algorithms perform binary mask (water, vegetation, building, etc.) and some methods are then applied to regularize and merge masks into a single multiclass mask.
<table border="0">
<tr>
<td>
<img src="docs/source/images/example_step0_PHR_image.png" alt="Initial VHR image" title="Initial VHR image"  width="80%">
</td>
<td>
<img src="docs/source/images/example_step1_watermask.png" alt="Water mask" title="Water mask"  width="80%">
</td>
<td>
<img src="docs/source/images/example_step2_vegetationmask.png" alt="Low/High vegetation mask" title="Low/High vegetation mask"  width="80%">
</td>
<td>
<img src="docs/source/images/example_step5_stack_regul.png" alt="Final mask" title="Final mask"  width="80%">
</td>
</tr>
</table>

## Install
You need to clone the repository and pip install SLUM.
```
git clone git@gitlab.cnes.fr:pluto/slum.git
```
To install SLUM, you need OTB and some libraries already installed on VRE OT.
Otherwise, if you are are connected to HAL, or working on your personal computer (Linux), 
you may set the environment as mentioned below.
### Create a virtual env with all libraries (if you don't use VRE OT)
On HAL, connect to a computing node to create & compile the virtual environment (needed to compile rasterio at install time)
```
qsub -l select=1:ncpus=8 -l walltime=01:00:00 -I
```
Load OTB and create a virtual env with some Python libraries
```
module load otb/7.4-python3.8.4
# Creates a virtual env base on Python 3.8.4
python -m venv slum_env
. slum_env/bin/activate
# upgrade pip and install several libraries
pip install pip --upgrade
pip install scikit-image scikit-learn geopandas scikit-learn-intelex matplotlib cython psutil
# install and compile rasterio with compatible GDAL 
pip install rasterio --no-binary :all:
```
### Pip install
Go to the directory where SLUM had been cloned and pip install it.
```
cd slum
pip install .
```
Your environment is ready, you can compute SLUM masks with slum_watermask, slum_urbanmask, etc.

## Use SLUM on HAL
On HAL, you can directly use SLUM by sourcing the following environment.
```
source /softs/projets/pluto/init_slum_v2.sh
```
This will load OTB 8.1-python3.8.4.

You can also use a .pbs script to launch different masks algorithms on your images.
```
qsub -v "PHR_IM=/work/scratch/tanguyy/public/RemyMartin/PHR_image_uint16.tif,OUTPUT_DIR=/work/scratch/tanguyy/public/RemyMartin/" /softs/projets/pluto/demo_slum/compute_all_masks.pbs
```
Two scripts (to calculate all the masks and the scores) are available in conf/ directory.


## Features

### Water mask
Water model is learned from Peckel (Global Surface Water) reference data and is based on NDVI/NDWI2 indices. 
Then the predicted mask is cleaned with Peckel, possibly with HAND or OSM maps and post-processed to clean artefacts.
```
slum_watermask <VHR input image> <your watermask.tif>
```
Type `slum_watermask -h` for complete list of options :

- bands identification (-red <1/3>), 
- add other raster features (-layers layer1 [layer 2 ..]), 
- add other raster filters (-filters file1 [file2 ..])
- post-process mask (-remove_small_holes, -binary_closing, etc.),
- saving of intermediate files (-save),
- etc.
### Vegetation mask
Vegetation mask are computed with an unsupervised clustering algorithm. First some primitives are computed from VHR image (NDVI, NDWI2, textures).
Then a segmentation is processed (SLIC, Large Scale Mean Shift, Felzenswalb) and segments are dispatched in several clusters depending
on their features.
A final labellisation affects a class to each segment (ie : high NDVI and low texture denotes for low vegetation).
```
slum_vegetationmask <VHR input image> <your vegetation mask.tif/.shp/.geojson/.gpkg>
```
Type `slum_vegetationmask -h` for complete list of options : 

- red/NIR bands
- segmentation mode and parameter for SLIC or Felzenswalb algorithms
- spectral threshold for texture (Structural Feature Set) computation
- number of workers (parallel processing for primitives and segmentation tasks)
- number of clusters affected to vegetation (3 by default - 33%)
- etc.


### Urban (building) mask
An urban model (building) is learned from WSF reference map. Adding an other OSM ground truth or water and vegetation masks improves model (by learning counter-example) and thus eliminates a lot of false positive detection. Then the predicted mask is regularized with a watershed algorithm, post-processed to clean artefacts and possibly cleaned with WSF to identify false positives.
The resulting mask is supposed to be stack with other masks (water, vegetation) to improve final rendering.
```
slum_urbanmask <VHR input image> <your urban mask>
```
Type `slum_urbanmask -h` for complete list of options :

- bands identification (-red <1/3>), 
- elimination of pixels identified as water or vegetation (-watermask <your watermask.tif>, -vegetationmask <your vegetationmask.tif>),
- post-process mask (-remove_small_holes, -binary_closing, -confidence_threshold, etc.), 
- identification of false positives (-remove_false_positive),
- saving of intermediate files (-save),
- etc.


### Shadow mask

### Stack all together

### Regularization step with Magiclip

### Quantify the quality of a mask

The predicted mask is compared to a given raster ground truth and some metrics such as the recall and the precision scores are calculated. The resulting mask shows the overlay of the prediction and the ground truth. An optional mode, useful for the urban mask, extracts the polygons of each raster and compare them, giving the number of expected buildings identified and the IoU score.
The analysis can be performed on a window of the input files.

```
slum_scores -im <predicted mask> -gt <raster ground truth - OSM, ..> -out <your overlay mask>
```

Type `slum_scores -h` for complete list of options :

- selection of a window (-startx, -starty, -sizex, -sizey),
- detection of the buildings (-polygonize) and iou score (-polygonize.union) with some parameters (-polygonize.area, -polygonize.unit, etc.),
- saving of intermediate files (-save)


## Documentation

Go in docs/ directory

## Contribution

See [Contribution](./CONTRIBUTING.md) manual

## References

This package was created with PLUTO-cookiecutter project template.


Inspired by [main cookiecutter template](https://github.com/audreyfeldroy/cookiecutter-pypackage) and 
[CARS cookiecutter template](https://gitlab.cnes.fr/cars/cars-cookiecutter)
