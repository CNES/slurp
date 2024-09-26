#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022-2024 CNES
#
# This file is part of slurp
#
""" Test vegetation mask with differents features and different arguments values"""

import pytest
import os
import glob

from tests.utils import get_output_path, get_aux_path

def write_command_compute_vegetationmask(nb_workers, valid_stack=None):
    output_image = get_output_path(pytest.features_test_img, "vegetationmask", remove=True)
    if valid_stack is None:
        valid_stack = get_aux_path(pytest.features_test_img, "valid_stack")
    
    return f"slurp_vegetationmask {pytest.main_config} -file_vhr {pytest.features_test_img} -n_workers {nb_workers} -vegetationmask {output_image} -valid {valid_stack} "

@pytest.mark.features
@pytest.mark.parametrize("non_veg_clusters",'null' )
def test_vegmask_max_value(non_veg_clusters):
    command = write_command_compute_urbanmask(1) + f"-non_veg_clusters {non_veg_clusters}"
    os.system(command)
    
def 