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
import sys

import pytest

import slurp.masks.shadowmask
import slurp.prepare.prepare
from tests.utils import get_aux_path, get_files_to_process, get_output_path
from tests.validation import validate_mask


# Input images
@pytest.fixture
def input_files(data_dir):
    return get_files_to_process("shadow", data_dir)


def prepare_shadowmask(main_config, output_dir, file, nb_workers):
    """Prepares the valid stack for shadow mask computation."""
    valid_stack = get_output_path(file, "valid_stack", output_dir, remove=True)
    command = (
        f"prepare.py {main_config} -file_vhr {file} -n_workers {nb_workers} -valid {valid_stack}"
    ).split()
    sys.argv = command
    slurp.prepare.prepare.main()
    assert os.path.exists(
        valid_stack
    ), f"The file {valid_stack} has not been created. Error during valid stack computation ?"
    return valid_stack


def compute_shadowmask(
    main_config, output_dir, ref_dir, file, nb_workers, valid_stack=None
):
    """Computes the shadow mask for a given image and validates output."""
    output_image = get_output_path(file, "shadowmask", output_dir, remove=True)
    if valid_stack is None:
        valid_stack = get_aux_path(file, "valid_stack", ref_dir)
    print(
        f"slurp_shadowmask {main_config} -file_vhr {file} -n_workers {nb_workers} "
        f"-shadowmask {output_image} -valid {valid_stack} -log_f"
    )
    command = (
        f"shadowmask.py {main_config} -file_vhr {file} -n_workers {nb_workers} "
        f"-shadowmask {output_image} -valid {valid_stack} -log_f"
    ).split()
    sys.argv = command
    slurp.masks.shadowmask.main()
    assert os.path.exists(
        output_image
    ), f"The file {output_image} has not been created. Error during shadowmask computation ?"
    return output_image


@pytest.mark.validation
def test_prepare_computation_and_validation_shadowmask(
    main_config, output_dir, ref_dir, input_files
):
    """Tests the full workflow of preparation, computation, and validation of shadow mask for each input file."""
    for input_file in input_files:
        valid_stack = prepare_shadowmask(
            main_config, output_dir, input_file, 1
        )
        validate_mask(valid_stack, "Prepare", ref_dir)
        output_image = compute_shadowmask(
            main_config, output_dir, ref_dir, input_file, 1, valid_stack
        )
        validate_mask(output_image, "Shadow", ref_dir)
