#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022-2024 CNES
#
# This file is part of slurp
#
""" Test prepare module with differents features and different arguments values"""

import pytest
import os
import glob

from tests.utils import get_output_path

def write_command_compute_prepare(nb_workers):
    valid_stack = get_output_path(pytest.features_test_img, "valid_stack", remove=True)
    ndvi = get_output_path(pytest.features_test_img, "ndvi", remove=True)
    ndwi = get_output_path(pytest.features_test_img, "ndwi", remove=True)
    texture = get_output_path(pytest.features_test_img, "texture", remove=True)
    
    return f"slurp_prepare {pytest.main_config} -file_vhr {pytest.features_test_img} -n_workers {nb_workers} -valid {valid_stack} -file_ndvi {ndvi} -file_ndwi {ndwi} -file_texture {texture} "
    
@pytest.mark.features
def test_absolute_analyse_glcm():
    command = write_command_compute_prepare(1) + f"-analyse_glcm True"
    os.system(command)
