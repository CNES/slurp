#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022-2024 CNES
#
# This file is part of slurp
#
""" Test shadow mask with differents features and different arguments values"""

import pytest
import os
import glob

from tests.utils import get_output_path

def write_command_compute_shadowmask(nb_workers, valid_stack=None):
    output_image = get_output_path(pytest.features_test_img, "shadowmask", remove=True)
    if valid_stack is None:
        valid_stack = get_aux_path(pytest.features_test_img, "valid_stack")
    
    return f"slurp_shadowmask {pytest.main_config} -file_vhr {pytest.features_test_img} -n_workers {nb_workers} -shadowmask {output_image} -valid {valid_stack} "

    
@pytest.mark.features
def test_absolute_threshold():
    command = write_command_compute_shadowmask(1) + f"-absolute_threshold True"
    os.system(command)
    
@pytest.mark.features
@pytest.mark.parametrize("percentile", [0,2,100])
def test_percentile(percentile):
    command = write_command_compute_shadowmask(1) + f"-percentile {percentile}"
    os.system(command)
    
@pytest.mark.features
@pytest.mark.parametrize("th_rgb,th_nir", [(0,0),(0.2,0.2)])
def test_percentile(th_rgb,th_nir):
    command = write_command_compute_shadowmask(1) + f"-th_nir {th_nir} -th_rgb {th_rgb}"
    os.system(command)