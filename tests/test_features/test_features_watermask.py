#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022-2024 CNES
#
# This file is part of slurp
#
""" Test water mask with differents features and different arguments values"""

import pytest
import os
import glob

from tests.utils import get_output_path, get_aux_path

def write_command_compute_watermask(nb_workers, valid_stack=None):
    output_image = get_output_path(pytest.features_test_img, "watermask", remove=True)
    if valid_stack is None:
        valid_stack = get_aux_path(pytest.features_test_img, "valid_stack")
    
    return f"slurp_watermask {pytest.main_config} -file_vhr {pytest.features_test_img} -n_workers {nb_workers} -watermask {output_image} -valid {valid_stack} "



    
@pytest.mark.features
def test_files_layers():
    command = write_command_compute_watermask(1) + f"-files_layers ['/work/datalake/static_aux/MASQUES/WSF/WSF2019_v1/WSF2019_v1.vrt']"
    os.system(command)

@pytest.mark.features
def test_hand_strict():
    command = write_command_compute_watermask(1) + f"-hand_strict True"
    os.system(command)
    
@pytest.mark.features
def test_simple_ndwi_threshold():
    command = write_command_compute_watermask(1) + f"-simple_ndwi_threshold True"
    os.system(command)
    
@pytest.mark.features
@pytest.mark.parametrize("samples_method", ['random','smart','grid'])
def test_samples_method(samples_method):
    command = write_command_compute_watermask(1) + f"-samples_method {samples_method}"
    os.system(command)
    
@pytest.mark.features
@pytest.mark.parametrize("nb_samples_water,nb_samples_other", [(10000,1000),(0,0)])
def test_nb_samples(nb_samples_water,nb_samples_other):
    command = write_command_compute_shadowmask(1) + f"-nb_samples_water {nb_samples_water} -nb_samples_other {nb_samples_other}"
    os.system(command)
    
@pytest.mark.features
def test_nb_samples_auto():
    command = write_command_compute_watermask(1) + f"-nb_samples_auto True"
    os.system(command)

@pytest.mark.features
def test_pekel_filter():
    command = write_command_compute_watermask(1) + f"-no_pekel_filter True"
    os.system(command)
    
@pytest.mark.features
def test_hand_filter():
    command = write_command_compute_watermask(1) + f"-hand_filter True"
    os.system(command)