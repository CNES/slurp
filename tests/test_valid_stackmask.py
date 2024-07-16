#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 Y T[3~[D.
#
# This file is part of slum
#
"""Tests for stack mask generation."""

import pytest
import os
import glob

from tests.utils import get_output_path, get_aux_path
from tests.validation import validate_mask


# Input images
input_files = glob.glob(os.path.join(pytest.data_dir, "all") + "/*.tif")

# Images to validate
predict_images = glob.glob(os.path.join(pytest.output_dir + "/stack_*.tif"))


def compute_stackmask(file, nb_workers):
    output_image = get_output_path(file, "stack", remove=True)
    
    masks_folder = os.path.join(pytest.data_dir, "stack", os.path.basename(file).replace('.tif', ''))
    watermask = os.path.join(masks_folder, "watermask.tif")
    vegetationmask = os.path.join(masks_folder, "vegetationmask.tif")
    urbanmask_proba = os.path.join(masks_folder, "urbanmask_proba.tif")
    shadowmask = os.path.join(masks_folder, "shadowmask.tif")
    wsf = os.path.join(masks_folder, "wsf.tif")
    valid_stack = get_aux_path(file, "valid_stack")
        
    os.system(f"slurp_stackmasks {pytest.main_config} -file_vhr {file} -n_workers {nb_workers} -stackmask {output_image} "
              f"-vegetationmask {vegetationmask} -watermask {watermask} -waterpred {watermask} "
              f"-urban_proba {urbanmask_proba} -shadow {shadowmask} -extracted_wsf {wsf} -valid {valid_stack} ")

    assert os.path.exists(output_image), f"The file {output_image} has not been created. Error during stackmask computation ?"
    return output_image


@pytest.mark.computation
@pytest.mark.parametrize("file", input_files)
def test_computation_stackmask(file):
    output_image = compute_stackmask(file, 1)


@pytest.mark.validation
@pytest.mark.parametrize("predict_file", predict_images)
def test_validation_stackmask(predict_file):
    validate_mask(predict_file, "Stack")


@pytest.mark.computation_and_validation
@pytest.mark.parametrize("file", input_files)
def test_computation_and_validation_stackask(file):
    output_image = compute_stackmask(file, 1)
    validate_mask(output_image, "Stack")
