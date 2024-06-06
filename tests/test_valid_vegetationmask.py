#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 Y T[3~[D.
#
# This file is part of slum
#
"""Tests for vegetationmask generation."""

import pytest
import os
import glob
from tests.utils import get_files_to_process, get_output_path, remove_file
from tests.validation import validate_mask

# Input images
input_files = get_files_to_process("vegetation")

# Images to validate
predict_images = glob.glob(os.path.join(pytest.output_dir + "/vegetationmask*.tif"))


def compute_vegetationmask(file):
    output_image = get_output_path(file, "vegetationmask")
    remove_file(output_image)
    os.system(f"slum_vegetationmask {file} -min_ndvi_veg 350 -max_ndvi_noveg 0 -remove_small_holes  50 -remove_small_objects 50 -binary_dilation 3 {output_image}") 
    assert os.path.exists(output_image) 
    return output_image


@pytest.mark.computation
@pytest.mark.parametrize("file", input_files)
def test_computation_vegetationmask(file):
    output_image = compute_vegetationmask(file)


@pytest.mark.validation
@pytest.mark.parametrize("predict_file", predict_images)
def test_validation_vegetationmask(predict_file):
    validate_mask(predict_file, "Vegetation", tolerance=True)

    
@pytest.mark.computation_and_validation
@pytest.mark.parametrize("file", input_files)
def test_computation_and_validation_vegetationmask(file):
    output_image = compute_vegetationmask(file)
    validate_mask(output_image, "Vegetation", tolerance=True)