#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022-2024 CNES
#
# This file is part of slurp
#
"""Test urban mask with differents features and different arguments values"""

import sys

import pytest

import slurp.masks.urbanmask
from tests.utils import get_aux_path, get_output_path


def write_command_compute_urbanmask(nb_workers, valid_stack=None):
    """Builds a command string to compute an urban mask using the urbanmask module."""
    output_image = get_output_path(
        pytest.features_test_img, "urbanmask", remove=True
    )
    if valid_stack is None:
        valid_stack = get_aux_path(pytest.features_test_img, "valid_stack")

    return f"urbanmask.py {pytest.main_config} -file_vhr {pytest.features_test_img} -n_workers {nb_workers} -urbanmask {output_image} -valid {valid_stack} "


@pytest.mark.features
@pytest.mark.parametrize("vegmask_min_value", [0, 21, 1000])
def test_vegmask_max_value(vegmask_min_value):
    """Tests the urban mask computation with different vegetation mask minimum values.
    vegmask_min_value: Vegetation min value for vegetated areas :
    all pixels with lower value will be predicted"""
    ndvi = get_output_path(pytest.features_test_img, 'ndvi')
    ndwi = get_output_path(pytest.features_test_img, 'ndwi')
    command = (
        write_command_compute_urbanmask(1)
        + f"-vegmask_min_value {vegmask_min_value} "
    ).split()
    sys.argv = command
    slurp.masks.urbanmask.main()


@pytest.mark.ci
@pytest.mark.parametrize("vegmask_min_value", [0, 21, 1000])
def test_vegmask_max_value_ci(vegmask_min_value):
    """Run the test test_vegmask_max_value with a specified valid_stack (for GithubCI)."""
    command = (
        write_command_compute_urbanmask(1, pytest.valid_stack)
        + f"-vegmask_min_value {vegmask_min_value}"
    ).split()
    sys.argv = command
    slurp.masks.urbanmask.main()


@pytest.mark.features
@pytest.mark.parametrize(
    "nb_samples_other,nb_samples_urban", [(0, 0), (5000, 1000)]
)
def test_nb_samples(nb_samples_other, nb_samples_urban):
    """Tests the urban mask computation with different sample counts for other and urban classes.
    nb_samples_other: Number of samples in other for learning.
    nb_samples_urban: Number of samples in buildings for learning"""
    command = (
        write_command_compute_urbanmask(1)
        + f"-nb_samples_other {nb_samples_other} -nb_samples_urban {nb_samples_urban}"
    ).split()
    sys.argv = command
    slurp.masks.urbanmask.main()


@pytest.mark.ci
@pytest.mark.parametrize(
    "nb_samples_other,nb_samples_urban", [(0, 0), (5000, 1000)]
)
def test_nb_samples_ci(nb_samples_other, nb_samples_urban):
    """Run test_nb_samples with a specified valid_stack (for GithubCI)."""
    command = (
        write_command_compute_urbanmask(1, pytest.valid_stack)
        + f"-nb_samples_other {nb_samples_other} -nb_samples_urban {nb_samples_urban}"
    ).split()
    sys.argv = command
    slurp.masks.urbanmask.main()


