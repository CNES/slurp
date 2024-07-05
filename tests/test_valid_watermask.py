#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 Y T[3~[D.
#
# This file is part of slurp
#
"""Tests for watermask generation."""

import pytest
import os
import glob
from tests.utils import get_files_to_process, get_output_path, get_aux_path
from tests.validation import validate_mask

# Input images
input_files = get_files_to_process("water")

# Images to validate
predict_images = glob.glob(os.path.join(pytest.output_dir + "/watermask*.tif"))


def prepare_watermask(file, nb_workers):
    valid_stack = get_output_path(file, "valid_stack", remove=True)
    ndvi = get_output_path(file, "ndvi", remove=True)
    ndwi = get_output_path(file, "ndwi", remove=True)
    pekel = get_output_path(file, "pekel", remove=True)
    hand = get_output_path(file, "hand", remove=True)
    global_pekel = "/work/datalake/static_aux/MASQUES/PEKEL/data2021/occurrence/occurrence.vrt"
    global_hand = "/work/datalake/static_aux/MASQUES/HAND_MERIT/hnd.vrt"
    
    os.system(f"slurp_prepare {pytest.main_config} -file_vhr {file} -n_workers {nb_workers} " 
              f"-valid_stack {valid_stack} -file_ndvi {ndvi} -file_ndwi {ndwi} " 
              f"-extracted_pekel {pekel} -extracted_hand {hand} -pekel {global_pekel} -hand {global_hand}")
    
    assert os.path.exists(valid_stack), f"The file {valid_stack} has not been created. Error during valid stack computation ?"
    assert os.path.exists(ndvi), f"The file {ndvi} has not been created. Error during NDVI computation ?"
    assert os.path.exists(ndwi), f"The file {ndwi} has not been created. Error during NDWI computation ?"
    assert os.path.exists(pekel), f"The file {pekel} has not been created. Error during Pekel extraction ?"
    assert os.path.exists(hand), f"The file {hand} has not been created. Error during HAND extraction ?"
    return valid_stack, ndvi, ndwi, pekel, hand


def compute_watermask(file, nb_workers):
    output_image = get_output_path(file, "watermask", remove=True)
    valid_stack = get_aux_path(file, "valid_stack")
    ndvi = get_aux_path(file, "ndvi")
    ndwi = get_aux_path(file, "ndwi")
    pekel = get_aux_path(file, "pekel")
    hand = get_aux_path(file, "hand")
    
    os.system(f"slurp_watermask {pytest.main_config} -file_vhr {file} -n_workers {nb_workers} "
              f"-watermask {output_image} -valid {valid_stack} -ndvi {ndvi} -ndwi {ndwi} -pekel {pekel} -hand {hand}")
    
    assert os.path.exists(output_image), f"The file {output_image} has not been created. Error during watermask computation ?"
    
    return output_image


@pytest.mark.prepare
@pytest.mark.parametrize("file", input_files)
def test_prepare_watermask(file):
    valid_stack, ndvi, ndwi, pekel, hand = prepare_watermask(file, 1)
    validate_mask(valid_stack, "Prepare")
    validate_mask(ndvi, "Prepare")
    validate_mask(ndwi, "Prepare")
    validate_mask(pekel, "Prepare")
    validate_mask(hand, "Prepare")


@pytest.mark.computation
@pytest.mark.parametrize("file", input_files)
def test_computation_watermask(file):
    output_image = compute_watermask(file, 1)


@pytest.mark.validation
@pytest.mark.parametrize("predict_file", predict_images)
def test_validation_watermask(predict_file):
    validate_mask(predict_file, "Water")

    
@pytest.mark.computation_and_validation
@pytest.mark.parametrize("file", input_files)
def test_computation_and_validation_watermask(file):
    output_image = compute_watermask(file, 1)
    validate_mask(output_image, "Water")
