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

import pytest

from tests.utils import get_aux_path, get_output_path
from tests.validation import validate_mask

# Input images
input_files = glob.glob(os.path.join(pytest.data_dir, "all") + "/*.tif")

# Images to validate
predict_images = glob.glob(os.path.join(pytest.output_dir + "/stack_*.tif"))


def compute_stackmask(file, nb_workers):
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

    os.system(
        f"slurp_stackmasks {pytest.main_config} -file_vhr {file} -n_workers {nb_workers} -stackmask {output_image} "
        f"-vegetationmask {vegetationmask} -watermask {watermask} "
        f"-urbanmask {urbanmask} -shadow {shadowmask} -wsf {wsf} -valid {valid_stack} -log_f"
    )

    assert os.path.exists(
        output_image
    ), f"The file {output_image} has not been created. Error during stackmask computation ?"
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
