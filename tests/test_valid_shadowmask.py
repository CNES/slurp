#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 Y T[3~[D.
#
# This file is part of slurp
#
"""Tests for shadowmask generation."""

import pytest
import os
import glob
from tests.utils import get_files_to_process, get_output_path, remove_file
from tests.validation import validate_mask

# Input images
input_files = get_files_to_process("shadow")

# Images to validate
predict_images = glob.glob(os.path.join(pytest.output_dir + "/shadowmask*.tif"))


def compute_shadowmask(file, nb_workers):
    output_image = get_output_path(file, "shadowmask")
    remove_file(output_image)
    os.system(f"slurp_shadowmask {file} -n_workers {nb_workers} -binary_opening 2 -remove_small_objects 100 -th_rgb 0.2 -th_nir 0.2 {output_image}") 
    assert os.path.exists(output_image) 
    return output_image


@pytest.mark.computation
@pytest.mark.parametrize("file", input_files)
def test_computation_shadowmask(file):
    output_image = compute_shadowmask(file, 1)


@pytest.mark.validation
@pytest.mark.parametrize("predict_file", predict_images)
def test_validation_shadowmask(predict_file):
    validate_mask(predict_file, "Shadow")

    
@pytest.mark.computation_and_validation
@pytest.mark.parametrize("file", input_files)
def test_computation_and_validation_shadowmask(file):
    output_image = compute_shadowmask(file, 1)
    validate_mask(output_image, "Shadow")