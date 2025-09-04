.. _tutorial:

SLURP tutorial
==============

SLURP is designed to compute a simple land cover map (water, low/high vegetation, bare groud, buildings) from a VHR (Very High Resolution) image with
4 bands (Red, Green, Blue, Near-Infrared). SLURP is based on few or unsupervised algorithms (random forest, clustering, segmentation) that need some auxiliary data for training step.

It had been validated with Pleiades and tested with WordView, CO3D, Pleiades NEO images. It shall also work with images with lower resolution (ex : SPOT 6/7).

This document presents an example of the SLURP pipeline on an image extract from Strasbourg (France).

.. figure:: _static/images/tutorials/strasbourg_vhr_image.png
   :alt: VHR image
   :width: 40%
   :align: center

   VHR image

-----------
First steps
-----------


Activate your environment containing SLURP
__________________________________________

.. warning::

    Depending on which computation you want to perform (cf. `SLURP pipeline <#slurp-pipeline>`_ section), you may need to install OTB on your system as an additional dependencies.
    Please refer to the OTB installation guide provided `here <https://www.orfeo-toolbox.org/CookBook-develop/Installation.html#create-an-healthy-python-environment-for-otb>`_  for more details.

.. code-block:: console

    $ source slurp_env/bin/activate


Get the necessary input data
----------------------------

In addition to the VHR input image, SLURP requires some auxiliary data to compute the different masks.

.. raw:: html

    <style>
      table.horizontal-borders th,
      table.horizontal-borders td {
        border-bottom: 1px solid #000;
        padding: 8px;
      }
    </style>

    <table class="horizontal-borders">
        <tr>
          <th>Auxiliary data</th>
          <th>Step / usage</th>
          <th>Comments</th>
        </tr>
        <tr>
          <td><a href="https://global-surface-water.appspot.com/download">Pekel</a> <br>(Global Surface Water - uint8)</td>
          <td><strong>Water mask prediction</strong> : Water occurrence [0–100] during the last 30 years, used to learn a water prediction model (<em>MANDATORY</em>)</td>
          <td>The global map is mandatory but you can also use some data by month</td>
        </tr>
        <tr>
          <td><a href="http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/">Hand MERIT</a> <br>(float32)</td>
          <td><strong>Water mask prediction</strong> : Map of height above nearest drainage used to optimize choice of "non water" samples in the training step (<em>OPTIONAL</em>)</td>
          <td>Free after registration (other kind of HAND maps exist)</td>
        </tr>
        <tr>
          <td><a href="https://download.geoservice.dlr.de/WSF2019/">WSF 2019</a> <br>(World Settlement Footprint - uint8)</td>
          <td><strong>Urban mask prediction</strong> : Global buildings map used to learn a building prediction model (<em>MANDATORY</em>)</td>
          <td>Could be replaced by a better resolution map if available (e.g., OSM buildings)</td>
        </tr>
        <tr>
          <td><a href="https://viewer.esa-worldcover.org/worldcover">ESA WorldCover</a> <br>(uint8)</td>
          <td><strong>Vegetation mask configuration</strong> : Global land cover map (10m resolution) used to customize vegetation clustering (<em>OPTIONAL</em>)</td>
          <td>Very helpful to parameterize balance between non-vegetation / low and high vegetation clusters (see vegetation mask algorithm)</td>
        </tr>
        <tr>
          <td><a href="https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM">Copernicus WBM</a> <br>(float32 or int16)</td>
          <td><strong>Stack mask</strong> : Water Body Mask from Copernicus, used to classify each water body from the watermask (<em>OPTIONAL</em>)</td>
          <td>If used, the final map will contain river, lake, sea classes</td>
        </tr>
    </table>
    <br><br>


Fill in the configuration file
------------------------------

A template for the configuration file is available `here <https://github.com/CNES/slurp/blob/main/conf/main_config.json>`_.
All parameters are explained in the `SLURP configuration <slurp_config.html>`_ page.

--------------
SLURP pipeline
--------------

Additional shell scripts are provided `here <https://github.com/CNES/slurp/tree/main/scripts>`_ to launch the steps of the
pipeline, but they can also be launched separately as explained below.

Usage without sensor mode (Additional OTB steps may be necessary)
_________________________________________________________________

If your input data has undergone geometric processing (e.g. orthorectification), you may need to use OTB to reproject and crop the
auxiliary data (Pekel, Hand and WSF masks) to match your input image.

.. note::

    **Pekel** and **Hand** data are used to compute the water mask, **WSF** data is used to compute the urban mask.
    Vegetation and shadow masks computation do not require any additional data.

    Therefore, **OTB is not required to compute vegetation and shadow masks.**

To install OTB, please refer to the `guide <https://www.orfeo-toolbox.org/CookBook-develop/Installation.html#create-an-healthy-python-environment-for-otb>`_ provided.

Once installed, you can use the OTB command line interface (CLI) to superimpose Pekel, Hand and WSF data with your input image.

.. code-block:: console

    # Superimpose Pekel, Hand and WSF with OTB
    # /!\ Adapt path depending on where your global Pekel database (resp. HAND, WSF) is located
    otbcli_Superimpose -inr path_to_input/input_file.tif -inm path_to_pekel/pekel_file.vrt -out "out/pekel.tif?&gdal:co:TILED=YES&gdal:co:COMPRESS=DEFLATE" uint8 -interpolator nn
    otbcli_Superimpose -inr path_to_input/input_file.tif -inm path_to_hand/hand_file.vrt -out "out/hand.tif?&gdal:co:TILED=YES&gdal:co:COMPRESS=DEFLATE"
    otbcli_Superimpose -inr path_to_input/input_file.tif -inm path_to_wsf/wsf_file.vrt -out "out/wsf.tif?&gdal:co:TILED=YES&gdal:co:COMPRESS=DEFLATE" uint8 -interpolator nn


Usage in sensor mode (OTB is not required)
------------------------------------------

If your input data image is in sensor geometry, please use SLURP with the ``sensor_mode`` option set to ``True``.
This mode allows to use images in sensor geometry, Pekel, Hand and WSF masks are directly processed during SLURP's preparation step.


SLURP's preparation step
------------------------

We need to prepare all the auxiliary data required to calculate the masks. This includes reprojection (when using ``sensor_mode`` option) and
cropping.

Run the following command :

.. code-block:: console

    slurp_prepare main_config.json

Or use the Python API :

.. code-block:: python

    import slurp-masks as slurp

    slurp_prepare(main_config = main_config.json)

The outputs of the prepare script have been stored to the `prepare` directory.

It contains :

    - the validity mask
    - the NDVI and NDWI primitives
    - the texture analysis
    - the superimposed Pekel, HAND and WSF files (only if using `sensor_mode` option)
    - the global config JSON file

.. raw:: html

    <table border="0" style="margin: auto; text-align: center;">
    <tr>
    <td style="width:300px;">
    <img src="_static/images/tutorials/ndvi.png" alt="NDVI image" title="NDVI image">
    </td>
    <td style="width:300px;">
    <img src="_static/images/tutorials/ndwi.png" alt="NDWI image" title="NDVI image">
    </td>
    <td style="width:300px;">
    <img src="_static/images/tutorials/texture.png" alt="Texture computation" title="Texture computation">
    </td>
    <td style="width:300px;">
    <img src="_static/images/tutorials/pekel.png" alt="Pekel mask" title="Pekel mask">
    </td>
    <td style="width:300px;">
    <img src="_static/images/tutorials/wsf.png" alt="WSF mask" title="WSF mask">
    </td>
    </tr>
    <tr>
    <td><b>NDVI of the VHR image</b></td>
    <td><b>NDWI of the VHR image</b></td>
    <td><b>Squared convolution with a kernel of ones</b></td>
    <td><b>Reprojected and cropped Pekel mask</b></td>
    <td><b>Reprojected and cropped WSF mask</b></td>
    </tr>
    </table>

SLURP's masks computation
-------------------------

Now, the different masks can be computed and finally stacked together. Use either the CLI or the Python API to launch each step.

**Generate the water mask**

.. code-block:: console

    slurp_watermask prepare/effective_used_config.json

.. code-block:: python

    import slurp-masks as slurp

    slurp_watermask(main_config = prepare/effective_used_config.json)


**Generate the vegetation mask**

.. code-block:: console

    slurp_vegetationmask prepare/effective_used_config.json

.. code-block:: python

    import slurp-masks as slurp

    slurp_vegetationmask(main_config = prepare/effective_used_config.json)

**Generate the shadow mask**

.. code-block:: console

    slurp_shadowmask prepare/effective_used_config.json

.. code-block:: python

    import slurp-masks as slurp

    slurp_shadowmask(main_config = prepare/effective_used_config.json)


**Generate the urban mask**

.. code-block:: console

    slurp_urbanmask prepare/effective_used_config.json

.. code-block:: python

    import slurp-masks as slurp

    slurp_urbanmask(main_config = prepare/effective_used_config.json)

**Stack the masks**

.. code-block:: console

    slurp_stackmasks prepare/effective_used_config.json

.. code-block:: python

    import slurp-masks as slurp

    slurp_stackmasks(main_config = prepare/effective_used_config.json)

The outputs masks have been stored to the `out` directory.

It contains :

    - the water mask
    - the shadow mask
    - the urban mask
    - the vegetation mask
    - the stack mask


Results
-------

.. raw:: html

    <table border="0" style="margin: auto; text-align: center;">
      <tr>
        <td style="width:300px;">
          <img src="_static/images/tutorials/watermask_without_otb.png" alt="Water mask" title="Water mask">
        </td>
        <td style="width:300px;">
          <img src="_static/images/tutorials/vegmask_without_otb.png" alt="Low/High vegetation and bare ground mask" title="Low/High vegetation mask">
        </td>
        <td style="width:300px;">
          <img src="_static/images/tutorials/shadowmask_without_otb.png" alt="Shadow mask" title="Shadow mask">
        </td>
        <td style="width:300px;">
          <img src="_static/images/tutorials/urbanmask_without_otb.png" alt="Urban probability" title="Urban probability">
        </td>
        <td style="width:300px;">
          <img src="_static/images/tutorials/stackmask_without_otb.png" alt="Final mask" title="Final mask">
        </td>
      </tr>
      <tr>
        <td><b>Water mask with style <code>conf/style_water.qml</code></b></td>
        <td><b>Vegetation mask with style <code>conf/style_vegetation.qml</code></b></td>
        <td><b>Shadow mask with style <code>conf/style_shadow.qml</code></b></td>
        <td><b>Urban mask (building probability)</b></td>
        <td><b>Stack mask with style <code>conf/style_stack.qml</code></b></td>
      </tr>
    </table>

