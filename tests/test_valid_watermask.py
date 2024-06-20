#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 Y T[3~[D.
#
# This file is part of slurp
#
"""Tests for watermask generation."""

import pytest
import os
import glob
from tests.utils import get_files_to_process, get_output_path, remove_file
from tests.validation import validate_mask

# Input images
input_files = get_files_to_process("water")

# Images to validate
predict_images = glob.glob(os.path.join(pytest.output_dir + "/watermask*.tif"))


def compute_watermask(file, nb_workers):
    output_image = get_output_path(file, "watermask")
    remove_file(output_image)
    os.system(f"slurp_watermask {pytest.main_config} -file_vhr {file} -n_workers {nb_workers} -watermask {output_image}") 
    assert os.path.exists(output_image), f"The file {output_image} has not been created. Error during watermask computation ?"
    return output_image


@pytest.mark.computation
@pytest.mark.parametrize("file", input_files)
def test_computation_watermask(file):
    output_image = compute_watermask(file, 1)


@pytest.mark.validation
@pytest.mark.parametrize("predict_file", predict_images)
def test_validation_watermask(predict_file):
    validate_mask(predict_file, "Water")

    
@pytest.mark.computation_and_validation
@pytest.mark.parametrize("file", input_files)
def test_computation_and_validation_watermask(file):
    output_image = compute_watermask(file, 1)
    validate_mask(output_image, "Water")
