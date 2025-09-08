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

"""Tests for vegetationmask generation."""

import glob
import os
import sys

import pytest

import slurp.masks.vegetationmask
import slurp.prepare.prepare
from tests.utils import get_aux_path, get_files_to_process, get_output_path
from tests.validation import validate_mask

# Input images
input_files = get_files_to_process("vegetation")


def prepare_vegetationmask(file, nb_workers):
    """Prepares the valid stack, NDVI, NDWI, and texture files for vegetation mask computation."""
    valid_stack = get_output_path(file, "valid_stack", remove=True)
    ndvi = get_output_path(file, "ndvi", remove=True)
    ndwi = get_output_path(file, "ndwi", remove=True)
    texture = get_output_path(file, "texture", remove=True)

    print(
        f"slurp_prepare {pytest.main_config} -file_vhr {file} -n_workers {nb_workers} "
        f"-valid {valid_stack} -file_ndvi {ndvi} -file_ndwi {ndwi} -file_texture {texture}"
    )
    command = (
        f"prepare.py {pytest.main_config} -file_vhr {file} -n_workers {nb_workers} "
        f"-valid {valid_stack} -file_ndvi {ndvi} -file_ndwi {ndwi} -file_texture {texture} -log_f"
    ).split()
    sys.argv = command
    slurp.prepare.prepare.main()

    assert os.path.exists(
        valid_stack
    ), f"The file {valid_stack} has not been created. Error during valid stack computation ?"
    assert os.path.exists(
        ndvi
    ), f"The file {ndvi} has not been created. Error during NDVI computation ?"
    assert os.path.exists(
        ndwi
    ), f"The file {ndwi} has not been created. Error during NDWI computation ?"
    assert os.path.exists(
        texture
    ), f"The file {texture} has not been created. Error during Texture computation ?"
    return valid_stack, ndvi, ndwi, texture


def compute_vegetationmask(
    file, nb_workers, valid_stack=None, ndvi=None, ndwi=None, texture=None
):
    """Computes the vegetation mask for a given image and validates output."""
    output_image = get_output_path(file, "vegetationmask", remove=True)
    if valid_stack is None:
        valid_stack = get_aux_path(file, "valid_stack")
    if ndvi is None:
        ndvi = get_aux_path(file, "ndvi")
    if ndwi is None:
        ndwi = get_aux_path(file, "ndwi")
    if texture is None:
        texture = get_aux_path(file, "texture")

    command = (
        f"vegetationmask.py {pytest.main_config} -file_vhr {file} -n_workers {nb_workers} "
        f"-vegetationmask {output_image} -valid {valid_stack} -ndvi {ndvi} -ndwi {ndwi} -texture {texture} -log_f"
    ).split()
    sys.argv = command
    slurp.masks.vegetationmask.main()

    assert os.path.exists(
        output_image
    ), f"The file {output_image} has not been created. Error during vegetationmask computation ?"
    return output_image


@pytest.mark.validation
@pytest.mark.parametrize("file", input_files)
def test_prepare_computation_and_validation_vegetationmask(file):
    """Tests the full workflow of preparation, computation, and validation of vegetation mask for each input file."""
    valid_stack, ndvi, ndwi, texture = prepare_vegetationmask(file, 1)
    validate_mask(valid_stack, "Prepare")
    validate_mask(ndvi, "Prepare")
    validate_mask(ndwi, "Prepare")
    validate_mask(texture, "Prepare")
    output_image = compute_vegetationmask(
        file, 1, valid_stack, ndvi, ndwi, texture
    )
    validate_mask(output_image, "Vegetation")
