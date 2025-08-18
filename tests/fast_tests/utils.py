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

"""Definition of global functions."""

import contextlib
import glob
import os

import pytest


def get_files_to_process(key):
    all_input_folder = os.path.join(pytest.data_dir, "all")
    key_input_folder = os.path.join(pytest.data_dir, key)
    return glob.glob(all_input_folder + "/*.tif") + glob.glob(
        key_input_folder + "/*.tif"
    )


def get_output_path(file, key, remove=False):
    assert os.path.exists(file), f"The file {file} doesn't exist"
    assert os.path.exists(
        pytest.output_dir
    ), f"The file {pytest.output_dir} doesn't exist"
    filename = os.path.basename(file)
    output_image = os.path.join(pytest.output_dir, key + "_" + filename)
    if remove:
        remove_file(output_image)
    return output_image


def get_aux_path(file, key):
    filename = os.path.basename(file)
    aux_image = os.path.join(
        pytest.ref_dir, "Prepare", "ref_" + key + "_" + filename
    )
    return aux_image


def remove_file(file):
    with contextlib.suppress(FileNotFoundError):
        os.remove(file)
