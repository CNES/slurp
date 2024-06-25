#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022-2024 CNES
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
    os.system(f"slurp_shadowmask {pytest.main_config} -file_vhr {file} -n_workers {nb_workers} -shadowmask {output_image}") 
    assert os.path.exists(output_image), f"The file {output_image} has not been created. Error during shadowmask computation ?"
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
