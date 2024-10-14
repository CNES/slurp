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
def test_vegmask_max_value():
    command = write_command_compute_vegetationmask(1) + f"-non_veg_clusters {True}"
    os.system(command)
    
@pytest.mark.features
def test_texture_mode():
    command = write_command_compute_vegetationmask(1) + f"-texture_mode 'no'"
    os.system(command)

@pytest.mark.features
@pytest.mark.parametrize("min_ndvi_veg,max_ndvi_noveg", [(0.5,2),(2,0.5)])
def test_percentile(min_ndvi_veg,max_ndvi_noveg):
    command = write_command_compute_vegetationmask(1) + f"-min_ndvi_veg {min_ndvi_veg} -max_ndvi_noveg {max_ndvi_noveg}"
    os.system(command)

@pytest.mark.features
@pytest.mark.parametrize("nb_clusters_veg,nb_clusters_low_veg", [(3,0),(0,5)])
def test_nb_clusters(nb_clusters_veg,nb_clusters_low_veg):
    command = write_command_compute_vegetationmask(1) + f"-nb_clusters_veg {nb_clusters_veg} -nb_clusters_low_veg {nb_clusters_low_veg}"
    os.system(command)
    
@pytest.mark.features
def test_max_low_veg():
    command = write_command_compute_vegetationmask(1) + f"-max_low_veg 3 "
    os.system(command)
    
@pytest.mark.features
def test_debug():
    command = write_command_compute_vegetationmask(1) + f"-debug True "
    os.system(command)    
