#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022-2024 CNES
#
# This file is part of slurp
#
"""Test shadow mask with differents features and different arguments values"""

import sys

import pytest

import slurp.masks.shadowmask
from tests.utils import get_aux_path, get_output_path


def write_command_compute_shadowmask(nb_workers, valid_stack=None):
    """Builds a command string to compute a shadow mask using the shadowmask module."""
    output_image = get_output_path(
        pytest.features_test_img, "shadowmask", remove=True
    )
    if valid_stack is None:
        valid_stack = get_aux_path(pytest.features_test_img, "valid_stack")

    return (
        f"shadowmask.py {pytest.main_config} "
        f"-file_vhr {pytest.features_test_img} "
        f"-n_workers {nb_workers} "
        f"-shadowmask {output_image} "
        f"-valid {valid_stack}"
    )


@pytest.mark.features
def test_absolute_threshold():
    """Tests the shadow mask computation with absolute thresholding enabled."""
    command = f"{write_command_compute_shadowmask(1)} -absolute_threshold 10.0".split()
    sys.argv = command
    slurp.masks.shadowmask.main()


@pytest.mark.ci
def test_absolute_threshold_ci():
    """Run the test_absolute_threshold test with a specified valid stack (for GithubCI)."""
    command = (
        write_command_compute_shadowmask(1, pytest.valid_stack)
        + " -absolute_threshold 10.0"
    ).split()
    sys.argv = command
    slurp.masks.shadowmask.main()


@pytest.mark.features
@pytest.mark.parametrize("percentile", [0, 2, 100])
def test_percentile(percentile):
    """Tests the shadow mask computation with different percentile values.
    The percentile value is used to cut histogram and estimate shadow threshold
    """
    command = f"{write_command_compute_shadowmask(1)} -percentile {percentile}".split()
    sys.argv = command
    slurp.masks.shadowmask.main()


@pytest.mark.ci
@pytest.mark.parametrize("percentile", [0, 2, 100])
def test_percentile_ci(percentile):
    """Run the test_percentile with a specified valid_stack (for GithubCI)."""
    command = (
        write_command_compute_shadowmask(1, pytest.valid_stack)
        + f" -percentile {percentile}"
    ).split()
    sys.argv = command
    slurp.masks.shadowmask.main()


@pytest.mark.features
@pytest.mark.parametrize("th_rgb,th_nir", [(0, 0), (0.2, 0.2)])
def test_percentile_nir_rgb(th_rgb, th_nir):
    """Tests the shadow mask computation with different threshold values
    for the nir and rgb bands."""
    command = (
        write_command_compute_shadowmask(1)
        + f" -th_nir {th_nir} -th_rgb {th_rgb}"
    ).split()
    sys.argv = command
    slurp.masks.shadowmask.main()


@pytest.mark.ci
@pytest.mark.parametrize("th_rgb,th_nir", [(0, 0), (0.2, 0.2)])
def test_percentile_nir_rgb_ci(th_rgb, th_nir):
    """Run test_percentile_nir_rgb_ci with a specified valid_stack (for GithubCI)."""
    command = (
        write_command_compute_shadowmask(1, pytest.valid_stack)
        + f" -th_nir {th_nir} -th_rgb {th_rgb}"
    ).split()
    sys.argv = command
    slurp.masks.shadowmask.main()
