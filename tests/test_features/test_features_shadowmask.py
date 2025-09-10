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


def write_command_compute_shadowmask(
    nb_workers, main_config, features_test_img, output_dir, valid_stack=None
):
    """Builds a command string to compute a shadow mask using the shadowmask module."""
    output_image = get_output_path(
        features_test_img, "shadowmask", output_dir, remove=True
    )
    if valid_stack is None:
        valid_stack = get_aux_path(features_test_img, "valid_stack")

    return (
        f"shadowmask.py {main_config} "
        f"-file_vhr {features_test_img} "
        f"-n_workers {nb_workers} "
        f"-shadowmask {output_image} "
        f"-valid {valid_stack}"
    )


@pytest.mark.features
def test_absolute_threshold(main_config, features_test_img, output_dir):
    """Tests the shadow mask computation with absolute thresholding enabled."""
    command = f"{write_command_compute_shadowmask(1, main_config, features_test_img, output_dir)} -absolute_threshold 10.0".split()
    sys.argv = command
    slurp.masks.shadowmask.main()


@pytest.mark.ci
def test_absolute_threshold_ci(
    main_config, features_test_img, output_dir, valid_stack
):
    """Run the test_absolute_threshold test with a specified valid stack (for GithubCI)."""
    command = (
        write_command_compute_shadowmask(
            1, main_config, features_test_img, output_dir, valid_stack
        )
        + " -absolute_threshold 10.0"
    ).split()
    sys.argv = command
    slurp.masks.shadowmask.main()


@pytest.mark.features
@pytest.mark.parametrize("percentile", [0, 2, 100])
def test_percentile(percentile, main_config, features_test_img, output_dir):
    """Tests the shadow mask computation with different percentile values.
    The percentile value is used to cut histogram and estimate shadow threshold
    """
    command = f"{write_command_compute_shadowmask(1, main_config, features_test_img, output_dir)} -percentile {percentile}".split()
    sys.argv = command
    slurp.masks.shadowmask.main()


@pytest.mark.ci
@pytest.mark.parametrize("percentile", [0, 2, 100])
def test_percentile_ci(
    percentile, main_config, features_test_img, output_dir, valid_stack
):
    """Run the test_percentile with a specified valid_stack (for GithubCI)."""
    command = (
        write_command_compute_shadowmask(
            1, main_config, features_test_img, output_dir, valid_stack
        )
        + f" -percentile {percentile}"
    ).split()
    sys.argv = command
    slurp.masks.shadowmask.main()


@pytest.mark.features
@pytest.mark.parametrize("th_rgb,th_nir", [(0, 0), (0.2, 0.2)])
def test_percentile_nir_rgb(
    th_rgb, th_nir, main_config, features_test_img, output_dir
):
    """Tests the shadow mask computation with different threshold values
    for the nir and rgb bands."""
    command = (
        write_command_compute_shadowmask(
            1, main_config, features_test_img, output_dir
        )
        + f" -th_nir {th_nir} -th_rgb {th_rgb}"
    ).split()
    sys.argv = command
    slurp.masks.shadowmask.main()


@pytest.mark.ci
@pytest.mark.parametrize("th_rgb,th_nir", [(0, 0), (0.2, 0.2)])
def test_percentile_nir_rgb_ci(
    th_rgb, th_nir, main_config, features_test_img, output_dir, valid_stack
):
    """Run test_percentile_nir_rgb_ci with a specified valid_stack (for GithubCI)."""
    command = (
        write_command_compute_shadowmask(
            1, main_config, features_test_img, output_dir, valid_stack
        )
        + f" -th_nir {th_nir} -th_rgb {th_rgb}"
    ).split()
    sys.argv = command
    slurp.masks.shadowmask.main()
