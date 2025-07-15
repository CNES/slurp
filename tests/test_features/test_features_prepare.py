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
import random
import shutil
import subprocess

import pytest

from tests.utils import get_output_path


def write_command_compute_prepare(nb_workers):
    valid_stack = get_output_path(
        pytest.features_test_img, "valid_stack", remove=True
    )
    ndvi = get_output_path(pytest.features_test_img, "ndvi", remove=True)
    ndwi = get_output_path(pytest.features_test_img, "ndwi", remove=True)
    texture = get_output_path(pytest.features_test_img, "texture", remove=True)

    return f"slurp_prepare {pytest.main_config} -file_vhr {pytest.features_test_img} -n_workers {nb_workers} -valid {valid_stack} -file_ndvi {ndvi} -file_ndwi {ndwi} -file_texture {texture} "


@pytest.mark.features
def test_absolute_analyse_glcm():
    command = write_command_compute_prepare(1) + f"--analyse_glcm"
    result = subprocess.run(command.split(), capture_output=True, text=True)
    assert result.returncode == 0, f"Error : {result.stderr}"

@pytest.mark.features
def test_prepare_update_config():
    """
    test that the effective_used_config.json file created during slurp_prepare
    is correctly updated.
    """
    possible_size = [128, 256, 512, 1024, 2048, 4096, 8192]
    i = random.randint(0, len(possible_size))
    command = (
        write_command_compute_prepare(1) + "-tile_max_size " + str(possible_size[i])
    )
    current_dir = os.getcwd()
    effective_used_config = os.path.join(current_dir, "out/effective_used_config.json")
    command += " -effective_used_config " + effective_used_config
    result = subprocess.run(command.split(), capture_output=True, text=True)
    assert result.returncode == 0, f"Error : {result.stderr}"

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
