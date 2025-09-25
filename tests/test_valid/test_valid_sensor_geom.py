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

from tests.utils import get_output_path
from tests.validation import validate_mask
import slurp.prepare.prepare


# Input images in sensor geometry
@pytest.fixture
def input_images(sensor_geom_dir):
    return glob.glob(sensor_geom_dir + "/*.tif")


@pytest.fixture
def input_files(input_images):
    DTMs = ["/work/datalake/static_aux/MNT/SRTM_30_hgt/N43E001.hgt"]

    # Create correct object for parametrize loop
    return list(zip(input_images, DTMs))


# Images to validate
@pytest.fixture
def predict_pekel(output_dir):
    return glob.glob(os.path.join(output_dir + "/pekel*.tif"))


@pytest.fixture
def predict_wsf(output_dir):
    return glob.glob(os.path.join(output_dir + "/wsf*.tif"))


def prepare_sensor_geom(main_config, output_dir, file, dtm, nb_workers):
    """Prepares sensor geometry data and validates output files."""
    valid_stack = get_output_path(file, "valid_stack", output_dir, remove=True)
    ndvi = get_output_path(file, "ndvi", output_dir, remove=True)
    ndwi = get_output_path(file, "ndwi", output_dir, remove=True)
    wsf = "/work/CAMPUS/etudes/Masques_CO3D/Validation/Baseline/Sensor/ref_wsf_xt_Toulouse_PontsJumeaux.tif"
    pekel = "/work/CAMPUS/etudes/Masques_CO3D/Validation/Baseline/Sensor/ref_pekel_xt_Toulouse_PontsJumeaux.tif"

    if not wsf:
        raise Exception(
            "Please add a global wsf file in 'config_tests.json' to run this test"
        )

    command = (
        f"prepare.py {main_config} -file_vhr {file} -n_workers {nb_workers} "
        f"-valid {valid_stack} -file_ndvi {ndvi} -file_ndwi {ndwi} -pekel {pekel} "
        f"-extracted_pekel {pekel} -extracted_wsf {wsf} -wsf {wsf} -sensor_mode True -dtm {dtm} "
        f"--no_analyse_glcm"
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
        wsf
    ), f"The file {wsf} has not been created. Error during WSF extraction ?"
    assert os.path.exists(
        pekel
    ), f"The file {pekel} has not been created. Error during WSF extraction ?"

    return valid_stack, ndvi, ndwi, wsf, pekel


@pytest.mark.validation
def test_prepare_sensor_geom(main_config, output_dir, input_files):
    """Tests the sensor geometry preparation and validates output masks."""
    for file, dtm in input_files:
        _, _, _, wsf, pekel = prepare_sensor_geom(
            main_config, output_dir, file, dtm, 1
        )
        validate_mask(pekel, "Sensor", valid_pixels=False)
        validate_mask(wsf, "Sensor", valid_pixels=False)


@pytest.mark.validation
def test_validation_sensor_geom_pekel(predict_pekel):
    """Tests the validation of Pekel masks generated from sensor geometry."""
    validate_mask(predict_pekel, "Sensor", valid_pixels=False)


@pytest.mark.validation
def test_validation_sensor_geom_wsf(predict_wsf):
    """Tests the validation of WSF masks generated from sensor geometry."""
    validate_mask(predict_wsf, "Sensor", valid_pixels=False)
