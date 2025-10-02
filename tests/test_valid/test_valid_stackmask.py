#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of SLURP
# (see https://github.com/CNES/slurp).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for stack mask generation."""

import glob
import os
import sys

import pytest

import slurp.masks.stack_masks
from tests.utils import get_aux_path, get_output_path
from tests.validation import validate_mask

# Input images
input_files = glob.glob(os.path.join(pytest.data_dir, "all") + "/*.tif")


def compute_stackmask(file, nb_workers):
    """Computes the stack mask for a given image and validates output."""
    output_image = get_output_path(file, "stack", remove=True)

    masks_folder = os.path.join(
        pytest.data_dir, "stack", os.path.basename(file).replace(".tif", "")
    )
    watermask = os.path.join(masks_folder, "watermask.tif")
    vegetationmask = os.path.join(masks_folder, "vegetationmask.tif")
    urbanmask = os.path.join(masks_folder, "urbanmask.tif")
    shadowmask = os.path.join(masks_folder, "shadowmask.tif")
    wsf = os.path.join(masks_folder, "wsf.tif")
    valid_stack = get_aux_path(file, "valid_stack")

    command = (
        f"slurp_stackmasks {pytest.main_config} -file_vhr {file} -n_workers {nb_workers} -stackmask {output_image} "
        f"-vegetationmask {vegetationmask} -watermask {watermask} "
        f"-urbanmask {urbanmask} -shadow {shadowmask} -wsf {wsf} -valid {valid_stack} -log_f"
    ).split()
    sys.argv = command
    slurp.masks.stack_masks.main()

    assert os.path.exists(
        output_image
    ), f"The file {output_image} has not been created. Error during stackmask computation ?"
    return output_image


@pytest.mark.ci
def test_computation_stackmask_ci():
    """Tests the computation of stack mask in a CI environment using test input masks."""
    masks_folder = "tests/inputs"
    vegetationmask = os.path.join(masks_folder, "vegetationmask.tif")
    watermask = os.path.join(masks_folder, "watermask.tif")
    urbanmask = os.path.join(masks_folder, "urbanmask.tif")
    shadowmask = os.path.join(masks_folder, "shadowmask.tif")
    wsf = os.path.join(masks_folder, "wsf.tif")
    output_image = get_output_path(
        pytest.features_test_img, "stackmask", remove=True
    )
    command = (
        f"slurp_stackmasks {pytest.main_config} -file_vhr {pytest.features_test_img} -n_workers 1 -stackmask {output_image} "
        f"-vegetationmask {vegetationmask} -watermask {watermask} "
        f"-urbanmask {urbanmask} -shadow {shadowmask} -wsf {wsf} -valid {pytest.valid_stack} "
    ).split()
    sys.argv = command
    slurp.masks.stack_masks.main()

    assert os.path.exists(
        output_image
    ), f"The file {output_image} has not been created. Error during stackmask computation ?"


@pytest.mark.validation
@pytest.mark.parametrize("file", input_files)
def test_computation_and_validation_stackask(file):
    """Tests both computation and validation of stack mask for each input file."""
    output_image = compute_stackmask(file, 1)
    validate_mask(output_image, "Stack")
