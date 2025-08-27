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
    output_image = get_output_path(
        pytest.features_test_img, "vegetationmask", remove=True
    )
    if valid_stack is None:
        valid_stack = get_aux_path(pytest.features_test_img, "valid_stack")

    return f"vegetationmask.py {pytest.main_config} -file_vhr {pytest.features_test_img} -n_workers {nb_workers} -vegetationmask {output_image} -valid {valid_stack} "


@pytest.mark.ci
def test_vegetation_mask_ci():
    command = write_command_compute_vegetationmask(
        1, pytest.valid_stack
    ).split()
    sys.argv = command
    slurp.masks.vegetationmask.main()


@pytest.mark.features
def test_vegmask_max_value():
    command = (
        write_command_compute_vegetationmask(1) + f"-non_veg_clusters"
    ).split()
    sys.argv = command
    slurp.masks.vegetationmask.main()


@pytest.mark.features
def test_texture_mode():
    command = (
        write_command_compute_vegetationmask(1) + f"-texture_mode no"
    ).split()
    sys.argv = command
    slurp.masks.vegetationmask.main()


@pytest.mark.features
@pytest.mark.parametrize("min_ndvi_veg,max_ndvi_noveg", [(1, 2), (2, 1)])
def test_percentile(min_ndvi_veg, max_ndvi_noveg):
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
    command = (
        write_command_compute_vegetationmask(1)
        + f"-nb_clusters_veg {nb_clusters_veg} -nb_clusters_low_veg {nb_clusters_low_veg}"
    ).split()
    sys.argv = command
    slurp.masks.vegetationmask.main()


@pytest.mark.features
def test_max_low_veg():
    command = (
        write_command_compute_vegetationmask(1) + f"-nb_clusters_low_veg 3 "
    ).split()
    sys.argv = command
    slurp.masks.vegetationmask.main()


@pytest.mark.features
def test_debug():
    command = (write_command_compute_vegetationmask(1) + f"--debug").split()
    sys.argv = command
    slurp.masks.vegetationmask.main()
