# Getting Started

This document presents an example of the SLURP pipeline on an image extract from Strasbourg (France).

<div align="center"><img src="../docs/source/images/tutorials/strasbourg_vhr_image.png" alt="VHR image" title="VHR image"  width="40%"></div>


## Get the input data

### Example with OTB

_Prerequisite : you must have an OTB package installed in your environment._

1. Activate your environment containing SLURP

Main README of SLURP project explains how to install SLURP.

On TREX, you can connect to a computing node then source the following environment : 
```
sinter -A cnes_level2 -N 1 -c 8 --time=01:00:00 --mem=30G --pty bash
source /work/CAMPUS/users/tanguyy/PLUTO/slurp_demo/init_slurp.sh
```

2. Get and extract data samples

```
tar xzvf /work/CAMPUS/etudes/Masques_CO3D/Data/Tutorial/example_with_otb.tar.gz
```

The folder contains :
- the input image
- the JSON config file
- a shell script to launch all masks
- an `aux` folder containing larger extracts of Pekel, Hand, WSF and ESA WorldLandCover files.

3. Retrieve the default main config file

```
cp conf/main_config.json example_with_otb/
cd example_with_otb
```

4. Create the out folders

```
mkdir -p prepare
mkdir -p out
```

### Example without OTB

_Note : Without OTB the ESA WorldLandCover analysis is not performed_

1. Activate your environment containing SLURP

Main README of SLURP project explains how to install SLURP.

2. Get and extract data samples

```
tar xzvf /work/CAMPUS/etudes/Masques_CO3D/Data/Tutorial/example_without_otb.tar.gz
```

The folder contains :
- the input image
- the JSON config file
- a shell script
- an `aux` folder containing reprojected Pekel, HAND and WSF files corresponding to the input image

3. Retrieve the default main config file

```
cp conf/main_config.json example_without_otb/
cd example_without_otb
```

4. Create the out folders

```
mkdir -p prepare
mkdir -p out
```

## Prepare the auxiliary data

At first, we need to prepare all the auxiliary data required to calculate the masks. This includes reprojections and cropping.

1. Run the following command :
```
slurp_prepare main_config.json -user_config config.json
```

2. Go to `prepare` directory to get the outputs of the prepare script

It contains :
- the validity mask
- the NDVI and NDWI primitives
- the texture analysis
- the superimposed Pekel, HAND and WSF files (only in the case of the example with OTB)
- the global config JSON file (with GLCM results if OTB is installed)

<table border="0">
<tr>
<td>
<img src="../docs/source/images/tutorials/ndvi.png" alt="NDVI image" title="NDVI image"  width="100%">
</td>
<td>
<img src="../docs/source/images/tutorials/ndwi.png" alt="NDWI image" title="NDVI image"  width="100%">
</td>
<td>
<img src="../docs/source/images/tutorials/texture.png" alt="Texture computation" title="Texture computation"  width="90%">
</td>
<td>
<img src="../docs/source/images/tutorials/pekel.png" alt="Pekel mask" title="Pekel mask"  width="90%">
</td>
<td>
<img src="../docs/source/images/tutorials/wsf.png" alt="WSF mask" title="WSF mask"  width="90%">
</td>
</tr>
<tr>
<td>NDVI of the VHR image</td>
<td>NDWI of the VHR image</td>
<td>Squared convolution with a kernel of ones</td>
<td>Reprojected and cropped Pekel mask</td>
<td>Reprojected and cropped WSF mask</td>
</tr>
</table>

## Generate the masks

1. Generate the water mask
```
slurp_watermask prepare/effective_used_config.json
```

2. Generate the vegetation mask
```
slurp_vegetationmask prepare/effective_used_config.json
```

3. Generate the shadow mask
```
slurp_shadowmask prepare/effective_used_config.json
```

4. Generate the urban mask
```
slurp_urbanmask prepare/effective_used_config.json
```

5. Stack the masks
```
slurp_stackmasks prepare/effective_used_config.json
```

6. Go to `out` directory to get the output masks

It contains :
- the water mask
- the shadow mask
- the urban mask
- the vegetation mask
- the stack mask

**Results with OTB (GLCM analysis performed)**

<table border="0">
<tr>
<td>
<img src="../docs/source/images/tutorials/watermask_with_otb.png" alt="Water mask" title="Water mask"  width="80%">
</td>
<td>
<img src="../docs/source/images/tutorials/vegmask_with_otb.png" alt="Low/High vegetation and bare ground mask" title="Low/High vegetation mask"  width="70%">
</td>
<td>
<img src="../docs/source/images/tutorials/shadowmask_with_otb.png" alt="Shadow mask" title="Shadow mask"  width="75%">
</td>
<td>
<img src="../docs/source/images/tutorials/urbanmask_with_otb.png" alt="Urban probability" title="Urban probability"  width="100%">
</td>
<td>
<img src="../docs/source/images/tutorials/stackmask_with_otb.png" alt="Final mask" title="Final mask"  width="80%">
</td>
</tr>
<tr>
<td>Water mask with style `conf/style_water.qml`</td>
<td>Vegetation mask with style `conf/style_vegetation.qml`</td>
<td>Shadow mask with style `conf/style_shadow.qml`</td>
<td>Urban mask (building probability)</td>
<td>Stack mask with style `conf/style_stack.qml`</td>
</tr>
</table>

**Results without OTB (without GLCM analysis)**

<table border="0">
<tr>
<td>
<img src="../docs/source/images/tutorials/watermask_without_otb.png" alt="Water mask" title="Water mask"  width="80%">
</td>
<td>
<img src="../docs/source/images/tutorials/vegmask_without_otb.png" alt="Low/High vegetation and bare ground mask" title="Low/High vegetation mask"  width="70%">
</td>
<td>
<img src="../docs/source/images/tutorials/shadowmask_without_otb.png" alt="Shadow mask" title="Shadow mask"  width="75%">
</td>
<td>
<img src="../docs/source/images/tutorials/urbanmask_without_otb.png" alt="Urban probability" title="Urban probability"  width="100%">
</td>
<td>
<img src="../docs/source/images/tutorials/stackmask_without_otb.png" alt="Final mask" title="Final mask"  width="80%">
</td>
</tr>
<tr>
<td>Water mask with style `conf/style_water.qml`</td>
<td>Vegetation mask with style `conf/style_vegetation.qml`</td>
<td>Shadow mask with style `conf/style_shadow.qml`</td>
<td>Urban mask (building probability)</td>
<td>Stack mask with style `conf/style_stack.qml`</td>
</tr>
</table>
