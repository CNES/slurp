<div align="center">
  <a href="https://gitlab.cnes.fr/pluto/slurp"><img src="docs/source/images/logo_SLURP_256.png" alt="SLURP" title="SLURP"  width="20%"></a>

<h4>slurp</h4>

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

**SLURP** : **S**mart **L**and **U**se **M**asks

SLURP proposes different algorithms to perform Land Use/Land Cover masks, with few data. Several algorithms perform binary mask (water, vegetation, building, etc.) and some methods are then applied to regularize and merge masks into a single multiclass mask.
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
You need to clone the repository and pip install SLURP.
```
git clone git@gitlab.cnes.fr:pluto/slurp.git
```
To install SLURP, you need OTB, [EOScale](https://gitlab.cnes.fr/pluto/eoscale) and some libraries already installed on VRE OT.

Otherwise, if you are are connected to TREX, or working on your personal computer (Linux), 
you may set the environment as mentioned below.
### Create a virtual env with all libraries (if you don't use VRE OT)
On TREX, connect to a computing node to create & compile the virtual environment (needed to compile rasterio at install time)
```
unset SLURM_JOB_ID ; srun -A cnes_level2 -N 1 -n 8 --time=02:00:00 --mem=64G --x11 --pty bash
```
Load OTB and create a virtual env with some Python libraries. 
Compile and install EOScale and then SLURP
```
module load otb/9.0.0-python3.8
# Creates a virtual env base on Python 3.8.13
python -m venv slurp_env
. slurp_env/bin/activate
# upgrade pip and install several libraries
pip install pip --upgrade
cd <EOScale source folder>
pip install .
cd <SLURP source folder>
pip install .
# for validation tests
pip install pytest
```
Your environment is ready, you can compute SLURP masks with slurp_watermask, slurp_urbanmask, etc.

## Use SLURP on TREX
On TREX, you can directly use SLURP by sourcing the following environment.
```
source /work/CAMPUS/users/tanguyy/PLUTO/slurp_demo/init_slurp.sh
```
This will load OTB 9.0 and all Python dependencies

You can also use a .pbs script to launch different masks algorithms on your images.
```
qsub -v "PHR_IM=/work/scratch/tanguyy/public/RemyMartin/PHR_image_uint16.tif,OUTPUT_DIR=/work/scratch/tanguyy/public/RemyMartin/" /softs/projets/pluto/demo_slurp/compute_all_masks.pbs
```
Two scripts (to calculate all the masks and the scores) are available in conf/ directory.


## Features

### Water mask
Water model is learned from Peckel (Global Surface Water) reference data and is based on NDVI/NDWI2 indices. 
Then the predicted mask is cleaned with Peckel, possibly with HAND maps and post-processed to clean artefacts.
```
slurp_watermask <VHR input image> <your watermask.tif>
```
Type `slurp_watermask -h` for complete list of options :

- bands identification (-red <1/3>), 
- add other raster features (-layers layer1 [layer 2 ..]), 
- add other raster filters (-filters file1 [file2 ..])
- post-process mask (-remove_small_holes, -binary_closing, etc.),
- saving of intermediate files (-save),
- etc.
### Vegetation mask
Vegetation mask are computed with an unsupervised clustering algorithm. First some primitives are computed from VHR image (NDVI, NDWI2, textures).
Then a segmentation is processed (SLIC) and segments are dispatched in several clusters depending on their features.
A final labellisation affects a class to each segment (ie : high NDVI and low texture denotes for low vegetation).
```
slurp_vegetationmask <VHR input image> <your vegetation mask.tif>
```
Type `slurp_vegetationmask -h` for complete list of options : 

- red/NIR bands
- segmentation mode and parameter for SLIC algorithms
- number of workers (parallel processing for primitives and segmentation tasks)
- number of clusters affected to vegetation (3 by default - 33%)
- etc.


### Urban (building) mask
An urban model (building) is learned from WSF reference map. The algorithm can take into account water and vegetation masks in order to improve samples selection (non building pixels will be chosen outside WSF and outside water/vegetation masks). 
The output is a "building probability" layer ([0..100]) that can be used by the stack algorithm.
```
slurp_urbanmask <VHR input image> <your urban mask>
```
Type `slurp_urbanmask -h` for complete list of options :

- bands identification (-red <1/3>), 
- elimination of pixels identified as water or vegetation (-watermask <your watermask.tif>, -vegetationmask <your vegetationmask.tif>),
- etc.

### Shadow mask
Shadow mask detects dark areas (supposed shadows), based on two thresholds (RGB, NIR). 
A post-processing step removes small shadows, holes, etc. The resulting mask is a three-classes mask (no shadow, small shadow, big shadows). 
The big shadows can be used in the stack algorithm in the regularization step.
```
slurp_shadowmask <VHR input image> <your shadow mask>
```

### Stack and regularize buildings
The stack algorithm take into account all previous masks to produce a 6 classes mask (water, low vegetation, high vegetation, building, bare soil, other) and an auxilliary height layer (low / high / unknown). 
The algorithm can regularize urban mask with a watershed algorithm based on building probability and context of surrounding areas. This algorithm first computes a gradient on the image and fills a marker layer with known classes. Then a watershed step helps to adjust contours along gradient image, thus regularizing buildings shapes.
```
slurp_stackmasks <VHR input image> <your stack image> -vegmask vegetation/vegetationmask.tif -watermask water/watermask.tif -urbanmask urban/urbanmask_proba.tif  -shadow shadow/shadowmask.tif -wsf urban/wsf.tif -remove_small_objects 500 -binary_closing 3
```

### Quantify the quality of a mask

The predicted mask is compared to a given raster ground truth and some metrics such as the recall and the precision scores are calculated. The resulting mask shows the overlay of the prediction and the ground truth. An optional mode, useful for the urban mask, extracts the polygons of each raster and compare them, giving the number of expected buildings identified and the IoU score.
The analysis can be performed on a window of the input files.

```
slurp_scores -im <predicted mask> -gt <raster ground truth - OSM, ..> -out <your overlay mask>
```

Type `slurp_scores -h` for complete list of options :

- selection of a window (-startx, -starty, -sizex, -sizey),
- detection of the buildings (-polygonize) and iou score (-polygonize.union) with some parameters (-polygonize.area, -polygonize.unit, etc.),
- saving of intermediate files (-save)

## Tests

The project comes with a suite of unit and functional tests. All the tests are available in tests/ directory.

To run them, launch the command `pytest` in the root of the slurp project. To run tests on a specific mask, execute `pytest tests/<file_name>"`.

By default, the tests generate the masks and then validate them by comparing them with a reference. You can choose to only compute the masks with `pytest -m computation` or validate them with `pytest -m validation`

You can change the default configuration for the tests by modifying the JSON file "tests/config\_tests". 


## Documentation

Go in docs/ directory

## Contribution

See [Contribution](./CONTRIBUTING.md) manual

## References

This package was created with PLUTO-cookiecutter project template.


Inspired by [main cookiecutter template](https://github.com/audreyfeldroy/cookiecutter-pypackage) and 
[CARS cookiecutter template](https://gitlab.cnes.fr/cars/cars-cookiecutter)
