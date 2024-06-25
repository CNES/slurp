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

from tests.utils import get_files_to_process, get_output_path, remove_file
from tests.validation import validate_mask


# Input images
input_files = glob.glob(os.path.join(pytest.data_dir, "all") + "/*.tif")

# Images to validate
predict_images = glob.glob(os.path.join(pytest.output_dir + "/stack_*.tif"))


def compute_stackmask(file, nb_workers):
    output_image = get_output_path(file, "stack")
    remove_file(output_image)
    
    masks_folder = os.path.join(pytest.data_dir, "stack", os.path.basename(file).replace('.tif', ''))
    watermask = os.path.join(masks_folder, "watermask.tif")
    vegetationmask = os.path.join(masks_folder, "vegetationmask.tif")
    urbanmask_proba = os.path.join(masks_folder, "urbanmask_proba.tif")
    shadowmask = os.path.join(masks_folder, "shadowmask.tif")
    wsf = os.path.join(masks_folder, "wsf.tif")
        
    os.system(f"slurp_stackmasks {pytest.main_config} -file_vhr {file} -vegetationmask {vegetationmask} -watermask {watermask} -waterpred {watermask} -urban_proba {urbanmask_proba} -shadow {shadowmask} " \
              f"-extracted_wsf {wsf} -n_workers {nb_workers} -remove_small_objects 300  -binary_closing 3 -binary_opening 3 -remove_small_holes 300 -building_erosion 2 " \
              f"-bonus_gt 10 -malus_shadow 10 -stackmask {output_image}")

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
