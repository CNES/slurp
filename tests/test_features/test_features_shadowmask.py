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
    command = (
        write_command_compute_shadowmask(1) + " -absolute_threshold 10.0"
    ).split()
    sys.argv = command
    slurp.masks.shadowmask.main()


@pytest.mark.ci
def test_absolute_threshold_ci():
    command = (
        write_command_compute_shadowmask(1, pytest.valid_stack)
        + " -absolute_threshold 10.0"
    ).split()
    sys.argv = command
    slurp.masks.shadowmask.main()


@pytest.mark.features
@pytest.mark.parametrize("percentile", [0, 2, 100])
def test_percentile(percentile):
    command = (
        write_command_compute_shadowmask(1) + f" -percentile {percentile}"
    ).split()
    sys.argv = command
    slurp.masks.shadowmask.main()


@pytest.mark.ci
@pytest.mark.parametrize("percentile", [0, 2, 100])
def test_percentile_ci(percentile):
    command = (
        write_command_compute_shadowmask(1, pytest.valid_stack)
        + f" -percentile {percentile}"
    ).split()
    sys.argv = command
    slurp.masks.shadowmask.main()


@pytest.mark.features
@pytest.mark.parametrize("th_rgb,th_nir", [(0, 0), (0.2, 0.2)])
def test_percentile_nir_rgb(th_rgb, th_nir):
    command = (
        write_command_compute_shadowmask(1)
        + f" -th_nir {th_nir} -th_rgb {th_rgb}"
    ).split()
    sys.argv = command
    slurp.masks.shadowmask.main()


@pytest.mark.ci
@pytest.mark.parametrize("th_rgb,th_nir", [(0, 0), (0.2, 0.2)])
def test_percentile_nir_rgb_ci(th_rgb, th_nir):
    command = (
        write_command_compute_shadowmask(1, pytest.valid_stack)
        + f" -th_nir {th_nir} -th_rgb {th_rgb}"
    ).split()
    sys.argv = command
    slurp.masks.shadowmask.main()
