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


def write_command_compute_vegetationmask(nb_workers, valid_stack=None):
    """Builds a command string to compute a vegetation mask using the vegetationmask module."""
    output_image = get_output_path(
        pytest.features_test_img, "vegetationmask", remove=True
    )
    if valid_stack is None:
        valid_stack = get_aux_path(pytest.features_test_img, "valid_stack")

    return f"vegetationmask.py {pytest.main_config} -file_vhr {pytest.features_test_img} -n_workers {nb_workers} -vegetationmask {output_image} -valid {valid_stack} "


@pytest.mark.ci
def test_vegetation_mask_ci():
    """Run the vegetation mask with a specified valid_stack (for GithubCI)."""
    command = write_command_compute_vegetationmask(
        1, pytest.valid_stack
    ).split()
    sys.argv = command
    slurp.masks.vegetationmask.main()


@pytest.mark.features
def test_vegmask_max_value():
    """Tests the vegetation mask computation with non-vegetation clusters enabled.
    non_veg_clusters: Labelize each 'non vegetation cluster' as 0, 1, 2 (..)
    instead of single label (0)"""
    command = (
        f"{write_command_compute_vegetationmask(1)}-non_veg_clusters".split()
    )
    sys.argv = command
    slurp.masks.vegetationmask.main()


@pytest.mark.features
def test_texture_mode():
    """ "Tests the vegetation mask computation with texture mode disabled.
    texture_mode: Labelize vegetation with (yes) or without (no) distinction low/high, "
    f"or get all NB_CLUSTERS vegetation clusters without distinction low/high.
    """
    command = (
        f"{write_command_compute_vegetationmask(1)}-texture_mode no".split()
    )
    sys.argv = command
    slurp.masks.vegetationmask.main()


@pytest.mark.features
@pytest.mark.parametrize("min_ndvi_veg,max_ndvi_noveg", [(1, 2), (2, 1)])
def test_percentile(min_ndvi_veg, max_ndvi_noveg):
    """Tests the vegetation mask computation with different NDVI thresholds
    for vegetation and non-vegetation.
    min_ndvi_veg: Minimal mean NDVI value to consider a cluster as vegetation (overload nb clusters choice).
    max_ndvi_noveg: Maximal mean NDVI value to consider a cluster as
    non-vegetation (overload nb clusters choice).
    """
    command = (
        write_command_compute_vegetationmask(1)
        + f"-min_ndvi_veg {min_ndvi_veg} -max_ndvi_noveg {max_ndvi_noveg}"
    ).split()
    sys.argv = command
    slurp.masks.vegetationmask.main()


@pytest.mark.features
@pytest.mark.parametrize(
    "nb_clusters_veg,nb_clusters_low_veg", [(3, 0), (0, 5)]
)
def test_nb_clusters(nb_clusters_veg, nb_clusters_low_veg):
    """Tests the vegetation mask computation with different
    numbers of clusters for vegetation and low vegetation.
    nb_cluster_veg: Nb of clusters considered as vegetation (1-NB_CLUSTERS).
    nb_clusters_low_veg: Nb of clusters considered as low vegetation(1-NB_CLUSTERS).
    """
    command = (
        write_command_compute_vegetationmask(1)
        + f"-nb_clusters_veg {nb_clusters_veg} -nb_clusters_low_veg {nb_clusters_low_veg}"
    ).split()
    sys.argv = command
    slurp.masks.vegetationmask.main()


@pytest.mark.features
def test_max_low_veg():
    """Tests the vegetation mask computation with a specified maximum value for low vegetation clusters.
    nb_clusters_low_veg: Nb of clusters considered as low vegetation(1-NB_CLUSTERS).
    """
    command = f"{write_command_compute_vegetationmask(1)}-nb_clusters_low_veg 3 ".split()
    sys.argv = command
    slurp.masks.vegetationmask.main()


@pytest.mark.features
def test_debug():
    """Tests the vegetation mask computation with debug mode enabled."""
    command = f"{write_command_compute_vegetationmask(1)}--debug".split()
    sys.argv = command
    slurp.masks.vegetationmask.main()
