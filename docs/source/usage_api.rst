=========
API Usage
=========

SLURP can be used as a Python library. Please start by installing it through pip.

.. code-block:: python

   pip install slurp-masks

Then you can import it in your Python scripts or notebooks.

.. code-block:: python

   import slurp-masks as slurp

Data preparation
----------------

Each mask needs some auxiliary files. They must be on the same
projection, resolution and bounding box of the VHR input image to enable
mask computation. You can generate this data yourself or use the prepare
script available in SLURP.

The prepare script enables :

    - Computation of stack validity (with or without a cloud mask)
    - Computation of NDVI and NDWI
    - Extraction of largest Pekel file
    - Extraction of largest HAND file
    - Extraction of WSF file
    - Computation of texture file with a convolution

It requires an OTB installation.

**To run the script**

1. Configure the JSON file. A template is available at
   conf/main_config.json with default values.
2. Update input, aux_layers, resources and prepare blocs inside the JSON
   file.
3. Use the python function:

.. code-block:: python

   slurp.slurp_prepare(main_config, overwrite, effective_used_config, logs_to_file, user_config,
                       file_vhr, sensor_mode, dtm, geoid_file, valid_stack, cloud_mask,
                       file_ndvi, file_ndwi, red, nir, green, pekel_method, pekel,
                       pekel_obs, pekel_monthly_occurrence, extracted_pekel, hand, extracted_hand,
                       wsf, extracted_wsf, file_texture, texture_rad, analyse_glcm,
                       land_cover_map, cropped_land_cover_map, n_workers, tile_max_size, multiproc_context)

Please check complete arguments description available `here <TODO>`_.
Beware that API arguments override the JSON arguments.

Features
--------

Water mask
~~~~~~~~~~

Water model is learned from Pekel (Global Surface Water) reference data
and is based on NDVI/NDWI2 indices. Then the predicted mask is cleaned
with Pekel, possibly with HAND (Height Above Nearest Drainage) maps and
post-processed to clean artefacts.

**To compute the mask**

1. Configure the JSON file : a template is available at
   conf/main_config.json with default values.
2. Update input, aux_layers and masks blocs inside the JSON file. To go
   further you can modify resources, post_process and water blocs.
3. Run the command :

.. code-block:: python

   slurp_watermask(main_config, debug, logs_to_file, user_config, file_vhr, valid_stack,
                   file_ndvi, file_ndwi, extracted_pekel, extracted_hand, files_layers,
                   file_filters, thresh_pekel, hand_strict, thresh_hand, strict_thresh,
                   save_mode, simple_ndwi_threshold, ndwi_threshold,
                   samples_method, nb_samples_water, nb_samples_other, nb_samples_auto,
                   auto_pct, smart_area_pct, smart_minimum, grid_spacing,
                   max_depth, nb_estimators, n_jobs,
                   no_pekel_filter, hand_filter, binary_closing,
                   area_closing, remove_small_holes, watermask,
                   value_classif, n_workers, tile_max_size, multiproc_context)


Please check complete arguments description available `here <TODO>`_.
Beware that API arguments override the JSON arguments.

Vegetation mask
~~~~~~~~~~~~~~~

Vegetation mask are computed with an unsupervised clustering algorithm.
First some primitives are computed from VHR image (NDVI, NDWI2,
textures). Then a segmentation is processed (SLIC) and segments are
dispatched in several clusters depending on their features. A final
labellisation affects a class to each segment (ie : high NDVI and low
texture denotes for low vegetation).

**To compute the mask**

1. Configure the JSON file : a template is available at
   conf/main_config.json with default values.
2. Update input, aux_layers and masks blocs inside the JSON file. To go
   further you can modify resources and vegetation blocs.
3. Run the command :

.. code-block:: python

   slurp_vegetationmask(main_config, debug, logs_to_file, user_config, file_vhr, valid_stack,
                        file_ndvi, file_ndwi, file_texture, texture_mode, filter_texture, save_mode,
                        slic_seg_size, slic_compactness, nb_clusters_veg, min_ndvi_veg,
                        max_ndvi_noveg, non_veg_clusters, nb_clusters_low_veg, max_texture_th,
                        binary_dilation, remove_small_objects, remove_small_holes,
                        vegetationmask, n_workers, tile_max_size, multiproc_context)

Please check complete arguments description available `here <TODO>`_.
Beware that API arguments override the JSON arguments.


Urban (building) mask
~~~~~~~~~~~~~~~~~~~~~

An urban model (building) is learned from WSF reference map. The
algorithm can take into account water and vegetation masks in order to
improve samples selection (non building pixels will be chosen outside
WSF and outside water/vegetation masks). The output is a “building
probability” layer ([0..100]) that can be used by the stack algorithm.

**To compute the mask**

1. Configure the JSON file : a template is available at
   conf/main_config.json with default values.
2. Update input, aux_layers and masks blocs inside the JSON file. To go
   further you can modify resources and urban blocs.
3. Run the command :

.. code-block:: python

   slurp_urbanmask(main_config, logs_to_file, user_config, file_vhr,
                   valid_stack, file_ndvi, file_ndwi, extracted_wsf, files_layers, watermask,
                   vegetationmask, shadowmask, vegmask_min_value, veg_binary_dilation, value_classif,
                   gt_binary_erosion, save_mode, nb_samples_urban, nb_samples_other, max_depth,
                   nb_estimators, n_jobs, urbanmask, n_workers, tile_max_size, multiproc_context)

Please check complete arguments description available `here <TODO>`_.
Beware that API arguments override the JSON arguments.


Shadow mask
~~~~~~~~~~~

Shadow mask detects dark areas (supposed shadows), based on two
thresholds (RGB, NIR). A post-processing step removes small shadows,
holes, etc. The resulting mask is a three-classes mask (no shadow, small
shadow, big shadows). The big shadows can be used in the stack algorithm
in the regularization step.

**To compute the mask**

1. Configure the JSON file : a template is available at
   conf/main_config.json with default values.
2. Update input, aux_layers and masks blocs inside the JSON file. To go
   further you can modify resources, post_process and shadow blocs.
3. Run the command :

.. code-block:: python

   slurp_shadowmask(main_config, logs_to_file, user_config, file_vhr, valid_stack, watermask,
                    th_rgb, th_nir, absolute_threshold, percentile, binary_opening,
                    remove_small_objects, shadowmask, n_workers, tile_max_size, multiproc_context)

Please check complete arguments description available `here <TODO>`_.
Beware that API arguments override the JSON arguments.

Stack and regularize buildings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The stack algorithm take into account all previous masks to produce a 6
classes mask (water, low vegetation, high vegetation, building, bare
soil, other) and an auxilliary height layer (low / high / unknown). The
algorithm can regularize urban mask with a watershed algorithm based on
building probability and context of surrounding areas. This algorithm
first computes a gradient on the image and fills a marker layer with
known classes. Then a watershed step helps to adjust contours along
gradient image, thus regularizing buildings shapes.

**To compute the mask**

1. Configure the JSON file : a template is available at
   conf/main_config.json with default values.
2. Update input, aux_layers and masks element inside the JSON file. To
   go further you can modify resources, post_process and stack blocs.
3. Run the command :

.. code-block:: python

   slurp_stackmasks(main_config, logs_to_file, user_config, file_vhr,
                    valid_stack, vegetationmask, watermask, urbanmask, shadowmask,
                    extracted_wsf, building_threshold, building_erosion, bonus_gt,
                    malus_shadow, stackmask, value_classif_low_veg, value_classif_high_veg,
                    value_classif_water, value_classif_buildings, value_classif_bare_ground,
                    value_classif_false_positive_buildings, value_classif_background, n_workers, tile_max_size,
                    multiproc_context)

Please check complete arguments description available `here <TODO>`_.
Beware that API arguments override the JSON arguments.


Contribution
------------

See `Contribution <./CONTRIBUTING.md>`__ manual

References
----------

This package was created with PLUTO-cookiecutter project template.

Inspired by `main cookiecutter
template <https://github.com/audreyfeldroy/cookiecutter-pypackage>`__
and `CARS cookiecutter
template <https://gitlab.cnes.fr/cars/cars-cookiecutter>`__

.. |Python| image:: https://img.shields.io/badge/python-v3.8+-blue.svg
   :target: https://www.python.org/downloads/release/python-380/

