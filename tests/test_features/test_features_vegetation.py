#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022-2024 CNES
#
# This file is part of slurp
#
"""Test vegetation mask with differents features and different arguments values"""

import sys

import pytest

import slurp.masks.vegetationmask
from tests.utils import get_aux_path, get_output_path


def write_command_compute_vegetationmask(
    nb_workers,
    main_config,
    features_test_img,
    output_dir,
    ref_dir,
    valid_stack=None,
):
    """Builds a command string to compute a vegetation mask using the vegetationmask module."""
    output_image = get_output_path(
        features_test_img, "vegetationmask", output_dir, remove=True
    )
    if valid_stack is None:
        valid_stack = get_aux_path(features_test_img, "valid_stack", ref_dir)

    return (
        f"vegetationmask.py {main_config} "
        f"-file_vhr {features_test_img} "
        f"-n_workers {nb_workers} "
        f"-vegetationmask {output_image} "
        f"-valid {valid_stack}"
    )


@pytest.mark.ci
def test_vegetation_mask_ci(
    main_config, features_test_img, output_dir, ref_dir, valid_stack
):
    """Run the vegetation mask with a specified valid_stack (for GithubCI)."""
    command = write_command_compute_vegetationmask(
        1, main_config, features_test_img, output_dir, ref_dir, valid_stack
    ).split()
    sys.argv = command
    slurp.masks.vegetationmask.main()


@pytest.mark.features
def test_vegmask_max_value(main_config, features_test_img, output_dir, ref_dir):
    """Tests the vegetation mask computation with non-vegetation clusters enabled.
    non_veg_clusters: Labelize each 'non vegetation cluster' as 0, 1, 2 (..)
    instead of single label (0)"""
    command = f"{write_command_compute_vegetationmask(1, main_config, features_test_img, output_dir, ref_dir)} -non_veg_clusters".split()
    sys.argv = command
    slurp.masks.vegetationmask.main()


@pytest.mark.features
def test_texture_mode(main_config, features_test_img, output_dir, ref_dir):
    """ "Tests the vegetation mask computation with texture mode disabled.
    texture_mode: Labelize vegetation with (yes) or without (no) distinction low/high, "
    f"or get all NB_CLUSTERS vegetation clusters without distinction low/high.
    """
    command = f"{write_command_compute_vegetationmask(1, main_config, features_test_img, output_dir, ref_dir)} -texture_mode no".split()
    sys.argv = command
    slurp.masks.vegetationmask.main()


@pytest.mark.features
@pytest.mark.parametrize("min_ndvi_veg,max_ndvi_noveg", [(1, 2), (2, 1)])
def test_percentile(
    min_ndvi_veg,
    max_ndvi_noveg,
    main_config,
    features_test_img,
    output_dir,
    ref_dir,
):
    """Tests the vegetation mask computation with different NDVI thresholds
    for vegetation and non-vegetation.
    min_ndvi_veg: Minimal mean NDVI value to consider a cluster as vegetation (overload nb clusters choice).
    max_ndvi_noveg: Maximal mean NDVI value to consider a cluster as
    non-vegetation (overload nb clusters choice).
    """
    command = (
        write_command_compute_vegetationmask(
            1, main_config, features_test_img, output_dir, ref_dir
        )
        + f" -min_ndvi_veg {min_ndvi_veg} -max_ndvi_noveg {max_ndvi_noveg}"
    ).split()
    sys.argv = command
    slurp.masks.vegetationmask.main()


@pytest.mark.features
@pytest.mark.parametrize(
    "labeling_strategy", ("nearest", "overestimate", "underestimate")
)
def test_autolabel(
    labeling_strategy,
    main_config,
    features_test_img,
    output_dir,
    ref_dir,
):
    """Tests the vegetation mask autolabel strategy with different parameters :
    choose the cluster repartition with nearest proportion to fit the input image, or overestimate
    vegetation rate, or underestimate.
    """
    command = (
        write_command_compute_vegetationmask(
            1, main_config, features_test_img, output_dir, ref_dir
        )
        + f" -autolabel -labeling_strategy {labeling_strategy}"
    ).split()
    sys.argv = command
    slurp.masks.vegetationmask.main()


@pytest.mark.features
@pytest.mark.parametrize(
    "nb_clusters_veg,nb_clusters_low_veg", [(3, 0), (0, 5)]
)
def test_nb_clusters(
    nb_clusters_veg,
    nb_clusters_low_veg,
    main_config,
    features_test_img,
    output_dir,
    ref_dir,
    valid_stack,
):
    """Tests the vegetation mask computation with different
    numbers of clusters for vegetation and low vegetation.
    nb_cluster_veg: Nb of clusters considered as vegetation (1-NB_CLUSTERS).
    nb_clusters_low_veg: Nb of clusters considered as low vegetation(1-NB_CLUSTERS).
    """
    command = (
        write_command_compute_vegetationmask(
            1, main_config, features_test_img, output_dir, ref_dir, valid_stack
        )
        + f" -nb_clusters_veg {nb_clusters_veg} -nb_clusters_low_veg {nb_clusters_low_veg}"
    ).split()
    sys.argv = command
    slurp.masks.vegetationmask.main()


@pytest.mark.features
def test_max_low_veg(main_config, features_test_img, output_dir, ref_dir):
    """Tests the vegetation mask computation with a specified maximum value for low vegetation clusters.
    nb_clusters_low_veg: Nb of clusters considered as low vegetation(1-NB_CLUSTERS).
    """
    command = f"{write_command_compute_vegetationmask(1, main_config, features_test_img, output_dir, ref_dir)} -nb_clusters_low_veg 3 ".split()
    sys.argv = command
    slurp.masks.vegetationmask.main()


@pytest.mark.features
def test_debug(main_config, features_test_img, output_dir, ref_dir):
    """Tests the vegetation mask computation with debug mode enabled."""
    command = f"{write_command_compute_vegetationmask(1, main_config, features_test_img, output_dir, ref_dir)} --debug".split()
    sys.argv = command
    slurp.masks.vegetationmask.main()
