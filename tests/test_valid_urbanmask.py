#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 Y T[3~[D.
#
# This file is part of slurp
#
"""Tests for urbanmask generation."""

import pytest
import os
import glob
from tests.utils import get_files_to_process, get_output_path, get_aux_path, remove_file
from tests.validation import validate_mask

# Input images
input_files = get_files_to_process("urban")

# Images to validate
predict_images = glob.glob(os.path.join(pytest.output_dir + "/urbanmask*_proba.tif"))


def prepare_urbanmask(file, nb_workers):
    valid_stack = get_output_path(file, "valid_stack", remove=True)
    ndvi = get_output_path(file, "ndvi", remove=True)
    ndwi = get_output_path(file, "ndwi", remove=True)
    wsf = get_output_path(file, "wsf", remove=True)
    global_wsf = "/work/datalake/static_aux/MASQUES/WSF/WSF2019_v1/WSF2019_cog.tif"
    
    os.system(f"slurp_prepare {pytest.main_config} -file_vhr {file} -n_workers {nb_workers} "
              f"-valid_stack {valid_stack} -file_ndvi {ndvi} -file_ndwi {ndwi} "
              f"-extracted_wsf {wsf} -wsf {global_wsf}")
    
    assert os.path.exists(valid_stack), f"The file {valid_stack} has not been created. Error during valid stack computation ?"
    assert os.path.exists(ndvi), f"The file {ndvi} has not been created. Error during NDVI computation ?"
    assert os.path.exists(ndwi), f"The file {ndwi} has not been created. Error during NDWI computation ?"
    assert os.path.exists(wsf), f"The file {wsf} has not been created. Error during WSF extraction ?"
    
    return valid_stack, ndvi, ndwi, wsf


def compute_urbanmask(file, nb_workers):
    output_image = get_output_path(file, "urbanmask")
    proba_image = output_image.replace(".tif", "_proba.tif")
    remove_file(proba_image)
    valid_stack = get_aux_path(file, "valid_stack")
    ndvi = get_aux_path(file, "ndvi")
    ndwi = get_aux_path(file, "ndwi")
    wsf = get_aux_path(file, "wsf")
    
    os.system(f"slurp_urbanmask {pytest.main_config} -file_vhr {file} -n_workers {nb_workers} -urbanmask {output_image} "
              f"-valid {valid_stack} -ndvi {ndvi} -ndwi {ndwi} -wsf {wsf}")
    
    assert os.path.exists(proba_image), f"The file {proba_image} has not been created. Error during urbanmask computation ?"
    
    return proba_image


@pytest.mark.prepare
@pytest.mark.parametrize("file", input_files)
def test_prepare_urbanmask(file):
    valid_stack, ndvi, ndwi, wsf = prepare_urbanmask(file, 1)
    validate_mask(valid_stack, "Prepare")
    validate_mask(ndvi, "Prepare")
    validate_mask(ndwi, "Prepare")
    validate_mask(wsf, "Prepare")


@pytest.mark.computation
@pytest.mark.parametrize("file", input_files)
def test_computation_urbanmask(file):
    output_image = compute_urbanmask(file, 1)


@pytest.mark.validation
@pytest.mark.parametrize("predict_file", predict_images)
def test_validation_urbanmask(predict_file):
    validate_mask(predict_file, "Urban", valid_pixels=False)

    
@pytest.mark.computation_and_validation
@pytest.mark.parametrize("file", input_files)
def test_computation_and_validation_urbanmask(file):
    output_image = compute_urbanmask(file, 1)
    validate_mask(output_image, "Urban", valid_pixels=False)
