#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022-2024 CNES
#
# This file is part of slurp
#
"""Test shadow mask with differents features and different arguments values"""

import subprocess

import pytest

from tests.utils import get_aux_path, get_output_path


def write_command_compute_shadowmask(nb_workers, valid_stack=None):
    output_image = get_output_path(
        pytest.features_test_img, "shadowmask", remove=True
    )
    if valid_stack is None:
        valid_stack = get_aux_path(pytest.features_test_img, "valid_stack")

    return (
        f"slurp_shadowmask {pytest.main_config} "
        f"-file_vhr {pytest.features_test_img} "
        f"-n_workers {nb_workers} "
        f"-shadowmask {output_image} "
        f"-valid {valid_stack}"
    )


@pytest.mark.features
def test_absolute_threshold():
    command = write_command_compute_shadowmask(1) + "-absolute_threshold 10"
    result = subprocess.run(command.split(), capture_output=True, text=True)
    assert result.returncode == 0, f"Error: {result.stderr}"


@pytest.mark.features
@pytest.mark.parametrize("percentile", [0, 2, 100])
def test_percentile(percentile):
    command = write_command_compute_shadowmask(1) + f"-percentile {percentile}"
    result = subprocess.run(command.split(), capture_output=True, text=True)
    assert result.returncode == 0, f"Error: {result.stderr}"


@pytest.mark.features
@pytest.mark.parametrize("th_rgb,th_nir", [(0, 0), (0.2, 0.2)])
def test_percentile_nir_rgb(th_rgb, th_nir):
    command = (
        write_command_compute_shadowmask(1)
        + f"-th_nir {th_nir} -th_rgb {th_rgb}"
    )
    result = subprocess.run(command.split(), capture_output=True, text=True)
    assert result.returncode == 0, f"Error: {result.stderr}"
