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

"""Tests for shadowmask generation."""

import glob
import os

import pytest

from tests.utils import get_aux_path, get_files_to_process, get_output_path
from tests.validation import validate_mask

# Input images
input_files = get_files_to_process("shadow")

# Images to validate
predict_images = glob.glob(os.path.join(pytest.output_dir + "/shadowmask*.tif"))


def prepare_shadowmask(file, nb_workers):
    valid_stack = get_output_path(file, "valid_stack", remove=True)
    os.system(
        f"slurp_prepare {pytest.main_config} -file_vhr {file} -n_workers {nb_workers} -valid {valid_stack} -log_f"
    )
    assert os.path.exists(
        valid_stack
    ), f"The file {valid_stack} has not been created. Error during valid stack computation ?"
    return valid_stack


def compute_shadowmask(file, nb_workers, valid_stack=None):
    output_image = get_output_path(file, "shadowmask", remove=True)
    if valid_stack is None:
        valid_stack = get_aux_path(file, "valid_stack")
    os.system(
        f"slurp_shadowmask {pytest.main_config} -file_vhr {file} -n_workers {nb_workers} "
        f"-shadowmask {output_image} -valid {valid_stack} -log_f"
    )
    assert os.path.exists(
        output_image
    ), f"The file {output_image} has not been created. Error during shadowmask computation ?"
    return output_image


@pytest.mark.prepare
@pytest.mark.parametrize("file", input_files)
def test_prepare_shadowmask(file):
    valid_stack = prepare_shadowmask(file, 1)
    validate_mask(valid_stack, "Prepare")


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


@pytest.mark.all
@pytest.mark.parametrize("file", input_files)
def test_prepare_computation_and_validation_shadowmask(file):
    valid_stack = prepare_shadowmask(file, 1)
    validate_mask(valid_stack, "Prepare")
    output_image = compute_shadowmask(file, 1, valid_stack)
    validate_mask(output_image, "Shadow")
