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

"""Tests for watermask generation."""

import os
import sys

import pytest

import slurp.masks.watermask
import slurp.prepare.prepare
from tests.utils import get_aux_path, get_files_to_process, get_output_path
from tests.validation import validate_mask


# Input images
@pytest.fixture
def input_files(data_dir):
    return get_files_to_process("water", data_dir)


def prepare_watermask(file, main_config, output_dir, pekel, hand, nb_workers):
    """Prepares the valid stack, NDVI, and NDWI files for water mask computation."""
    valid_stack = get_output_path(file, "valid_stack", output_dir, remove=True)
    ndvi = get_output_path(file, "ndvi", output_dir, remove=True)
    ndwi = get_output_path(file, "ndwi", output_dir, remove=True)

    command = (
        f"prepare.py {main_config} -file_vhr {file} -n_workers {nb_workers} "
        f"-valid {valid_stack} -file_ndvi {ndvi} -file_ndwi {ndwi} "
        f"-pekel {pekel} -hand {hand} -log_f"
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
    return valid_stack, ndvi, ndwi


def compute_watermask(
    file,
    main_config,
    output_dir,
    ref_dir,
    nb_workers,
    valid_stack=None,
    ndvi=None,
    ndwi=None,
    pekel=None,
    hand=None,
):
    """Computes the water mask for a given image and validates output."""
    output_image = get_output_path(file, "watermask", output_dir, remove=True)
    if valid_stack is None:
        valid_stack = get_aux_path(file, "valid_stack", ref_dir)
    if ndvi is None:
        ndvi = get_aux_path(file, "ndvi", ref_dir)
    if ndwi is None:
        ndwi = get_aux_path(file, "ndwi", ref_dir)
    if pekel is None:
        pekel = get_aux_path(file, "pekel", ref_dir)
    if hand is None:
        hand = get_aux_path(file, "hand", ref_dir)

    command = (
        f"watermask.py {main_config} -file_vhr {file} -n_workers {nb_workers} "
        f"-watermask {output_image} -valid {valid_stack} -ndvi {ndvi} -ndwi {ndwi} -pekel {pekel} -hand {hand} -log_f"
    ).split()
    sys.argv = command
    slurp.masks.watermask.main()

    assert os.path.exists(
        output_image
    ), f"The file {output_image} has not been created. Error during watermask computation ?"

    return output_image


@pytest.mark.validation
def test_prepare_computation_and_validation_watermask(
    input_files, main_config, output_dir, ref_dir, pekel, hand
):
    """Tests the full workflow of preparation, computation, and validation of water mask for each input file."""
    for input_file in input_files:
        valid_stack, ndvi, ndwi = prepare_watermask(
            input_file, main_config, output_dir, pekel, hand, 1
        )
        validate_mask(valid_stack, "Prepare", ref_dir)
        validate_mask(ndvi, "Prepare", ref_dir)
        validate_mask(ndwi, "Prepare", ref_dir)
        output_image = compute_watermask(
            input_file,
            main_config,
            output_dir,
            ref_dir,
            1,
            valid_stack,
            ndvi,
            ndwi,
        )
        validate_mask(output_image, "Water", ref_dir)
