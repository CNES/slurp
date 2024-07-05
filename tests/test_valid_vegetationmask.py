#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022-2024 CNES
#
# This file is part of slurp
#
"""Tests for vegetationmask generation."""

import pytest
import os
import glob
from tests.utils import get_files_to_process, get_output_path, get_aux_path
from tests.validation import validate_mask

# Input images
input_files = get_files_to_process("vegetation")

# Images to validate
predict_images = glob.glob(os.path.join(pytest.output_dir + "/vegetationmask*.tif"))


def prepare_vegetationmask(file, nb_workers):
    valid_stack = get_output_path(file, "valid_stack", remove=True)
    ndvi = get_output_path(file, "ndvi", remove=True)
    ndwi = get_output_path(file, "ndwi", remove=True)
    texture = get_output_path(file, "texture", remove=True)
    
    os.system(f"slurp_prepare {pytest.main_config} -file_vhr {file} -n_workers {nb_workers} -texture_rad 5 "
              f"-valid {valid_stack} -file_ndvi {ndvi} -file_ndwi {ndwi} -file_texture {texture}")
    
    assert os.path.exists(valid_stack), f"The file {valid_stack} has not been created. Error during valid stack computation ?"
    assert os.path.exists(ndvi), f"The file {ndvi} has not been created. Error during NDVI computation ?"
    assert os.path.exists(ndwi), f"The file {ndwi} has not been created. Error during NDWI computation ?"
    assert os.path.exists(texture), f"The file {texture} has not been created. Error during Texture computation ?"
    return valid_stack, ndvi, ndwi, texture


def compute_vegetationmask(file, nb_workers):
    output_image = get_output_path(file, "vegetationmask", remove=True)
    valid_stack = get_aux_path(file, "valid_stack")
    ndvi = get_aux_path(file, "ndvi")
    ndwi = get_aux_path(file, "ndwi")
    texture = get_aux_path(file, "texture")
    
    os.system(f"slurp_vegetationmask {pytest.main_config} -file_vhr {file} -n_workers {nb_workers} "
              f"-vegetationmask {output_image} -valid {valid_stack} -ndvi {ndvi} -ndwi {ndwi} -texture {texture}")
    
    assert os.path.exists(output_image), f"The file {output_image} has not been created. Error during vegetationmask computation ?"
    return output_image


@pytest.mark.prepare
@pytest.mark.parametrize("file", input_files)
def test_prepare_vegetationmask(file):
    valid_stack, ndvi, ndwi, texture = prepare_vegetationmask(file, 1)
    validate_mask(valid_stack, "Prepare")
    validate_mask(ndvi, "Prepare")
    validate_mask(ndwi, "Prepare")
    validate_mask(texture, "Prepare")


@pytest.mark.computation
@pytest.mark.parametrize("file", input_files)
def test_computation_vegetationmask(file):
    output_image = compute_vegetationmask(file, 1)


@pytest.mark.validation
@pytest.mark.parametrize("predict_file", predict_images)
def test_validation_vegetationmask(predict_file):
    validate_mask(predict_file, "Vegetation")

    
@pytest.mark.computation_and_validation
@pytest.mark.parametrize("file", input_files)
def test_computation_and_validation_vegetationmask(file):
    output_image = compute_vegetationmask(file, 1)
    validate_mask(output_image, "Vegetation")
