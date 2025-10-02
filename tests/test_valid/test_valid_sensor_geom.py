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

from tests.utils import get_output_path
from tests.validation import validate_mask

# Input images in sensor geometry
input_images = glob.glob(pytest.sensor_geom_dir + "/*.tif")
DTMs = ["/work/datalake/static_aux/MNT/SRTM_30_hgt/N43E001.hgt"]

# Create correct object for parametrize loop
input_files = []
for i in range(len(input_images)):
    input_files.append((input_images[i], DTMs[i]))

# Images to validate
predict_pekel = glob.glob(os.path.join(pytest.output_dir + "/pekel*.tif"))
predict_wsf = glob.glob(os.path.join(pytest.output_dir + "/wsf*.tif"))


def prepare_sensor_geom(file, dtm, nb_workers):
    """Prepares sensor geometry data and validates output files."""
    filename = os.path.basename(file)
    valid_stack = get_output_path(file, "valid_stack", remove=True)
    ndvi = get_output_path(file, "ndvi", remove=True)
    ndwi = get_output_path(file, "ndwi", remove=True)
    wsf = get_output_path(file, "wsf", remove=True)
    pekel = get_output_path(file, "pekel", remove=True)

    if not pytest.wsf:
        raise Exception(
            "Please add a global wsf file in 'config_tests.json' to run this test"
        )

    os.system(
        f"slurp_prepare {pytest.main_config} -file_vhr {file} -n_workers {nb_workers} "
        f"-valid {valid_stack} -file_ndvi {ndvi} -file_ndwi {ndwi} -pekel {pytest.pekel} -extracted_pekel {pekel} -extracted_wsf {wsf} -wsf {pytest.wsf} -sensor_mode True -dtm {dtm}"
    )

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
        wsf
    ), f"The file {wsf} has not been created. Error during WSF extraction ?"
    assert os.path.exists(
        pekel
    ), f"The file {pekel} has not been created. Error during WSF extraction ?"

    return valid_stack, ndvi, ndwi, wsf, pekel


@pytest.mark.validation
@pytest.mark.parametrize("file, dtm", input_files)
def test_prepare_sensor_geom(file, dtm):
    """Tests the sensor geometry preparation and validates output masks."""
    _, _, _, wsf, pekel = prepare_sensor_geom(file, dtm, 1)
    validate_mask(pekel, "Sensor", valid_pixels=False)
    validate_mask(wsf, "Sensor", valid_pixels=False)


@pytest.mark.validation
@pytest.mark.parametrize("file_pekel", predict_pekel)
def test_validation_sensor_geom_pekel(file_pekel):
    """Tests the validation of Pekel masks generated from sensor geometry."""
    validate_mask(file_pekel, "Sensor", valid_pixels=False)


@pytest.mark.validation
@pytest.mark.parametrize("file_wsf", predict_wsf)
def test_validation_sensor_geom_wsf(file_wsf):
    """Tests the validation of WSF masks generated from sensor geometry."""
    validate_mask(file_wsf, "Sensor", valid_pixels=False)
