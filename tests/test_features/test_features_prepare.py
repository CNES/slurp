#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022-2024 CNES
#
# This file is part of slurp
#
"""Test prepare module with differents features and different arguments values"""

import json
import os
import sys
import shutil
import subprocess
import random

import pytest

import slurp.prepare.prepare
from tests.utils import get_output_path


def write_command_compute_prepare(nb_workers, valid_stack=None):
    """Builds a command string to run the prepare module with
    specified worker count and valid stack."""
    if not valid_stack:
        valid_stack = get_output_path(
            pytest.features_test_img, "valid_stack", remove=True
        )
    ndvi = get_output_path(pytest.features_test_img, "ndvi")
    ndwi = get_output_path(pytest.features_test_img, "ndwi")
    texture = get_output_path(pytest.features_test_img, "texture")

    return f"prepare.py {pytest.main_config} -file_vhr {pytest.features_test_img} -n_workers {nb_workers} -valid {valid_stack} -file_ndvi {ndvi} -file_ndwi {ndwi} -file_texture {texture} "


@pytest.mark.features
def test_absolute_analyse_glcm():
    """Tests the prepare module with glcm analysis enabled
    glcm: Use a global land cover map to calculate the better number of
    vegetation cluster to use for mask computation"""
    command = f"{write_command_compute_prepare(1)}--analyse_glcm".split()
    sys.argv = command
    slurp.prepare.prepare.main()


@pytest.mark.ci
def test_absolute_analyse_glcm_ci():
    """Run the test_absolute_analyse_glcm with a specified valid_stack (for GithubCI)."""
    command = (
        write_command_compute_prepare(1, pytest.valid_stack)
        + "--no_analyse_glcm"
    ).split()
    sys.argv = command
    slurp.prepare.prepare.main()


@pytest.mark.features
def test_prepare_update_config():
    """
    test that the effective_used_config.json file created during slurp_prepare
    is correctly updated.
    """
    possible_size = [128, 256, 512, 1024, 2048, 4096, 8192]
    i = random.randint(0, len(possible_size) - 1)
    command = (
        write_command_compute_prepare(1)
        + "-tile_max_size "
        + str(possible_size[i])
    )
    current_dir = os.getcwd()
    effective_used_config = os.path.join(
        current_dir, "out/effective_used_config.json"
    )
    command += f" -effective_used_config {effective_used_config}"
    command = command.split()
    sys.argv = command
    slurp.prepare.prepare.main()
    with open(effective_used_config, "r", encoding="utf8") as json_file:
        config = json.load(json_file)
        for key in config:
            for sub_key in config[key]:
                if sub_key == "tile_max_size":
                    assert config[key][sub_key] == possible_size[i]
                    break

    dir_to_remove = os.path.join(current_dir, "out")
    if os.path.exists(dir_to_remove):
        shutil.rmtree(dir_to_remove)

