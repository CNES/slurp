#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 Y T[3~[D.
#
# This file is part of slurp
#
"""Tests for urbanmask generation."""

import pytest
import os
import glob
from tests.utils import get_files_to_process, get_output_path, remove_file
from tests.validation import validate_mask

# Input images
input_files = get_files_to_process("urban")

# Images to validate
predict_images = glob.glob(os.path.join(pytest.output_dir + "/urbanmask*_proba.tif"))


def compute_urbanmask(file, nb_workers):
    output_image = get_output_path(file, "urbanmask")
    proba_image = output_image.replace(".tif", "_proba.tif")
    remove_file(proba_image)
    os.system(f"slurp_urbanmask {file} -n_workers {nb_workers} -remove_false_positive -remove_small_objects 400 -remove_small_holes 50 " \
              f"-binary_closing 3 -binary_opening 3 {output_image}")
    assert os.path.exists(proba_image) 
    return proba_image


@pytest.mark.computation
@pytest.mark.parametrize("file", input_files)
def test_computation_urbanmask(file):
    output_image = compute_urbanmask(file, 1)


@pytest.mark.validation
@pytest.mark.parametrize("predict_file", predict_images)
def test_validation_urbanmask(predict_file):
    validate_mask(predict_file, "Urban", valid_pixels=False)

    
@pytest.mark.computation_and_validation
@pytest.mark.parametrize("file", input_files)
def test_computation_and_validation_urbanmask(file):
    output_image = compute_urbanmask(file, 1)
    validate_mask(output_image, "Urban", valid_pixels=False)