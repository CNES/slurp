#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022-2024 CNES
#
# This file is part of slurp
#
"""Test prepare module with differents features and different arguments values"""

import glob
import os
import sys

import pytest

import slurp.prepare.prepare
from test_low_resolution.utils import get_output_path


def write_command_compute_prepare(nb_workers):
    ndvi = get_output_path(pytest.features_test_img, "ndvi", remove=True)
    ndwi = get_output_path(pytest.features_test_img, "ndwi", remove=True)
    texture = get_output_path(pytest.features_test_img, "texture", remove=True)

    return f"prepare.py {pytest.main_config} -file_vhr {pytest.features_test_img} -n_workers {nb_workers} -valid {pytest.valid_stack} -file_ndvi {ndvi} -file_ndwi {ndwi} -file_texture {texture} --no_analyse_glcm"


@pytest.mark.fast
def test_absolute_analyse_glcm():
    command = (write_command_compute_prepare(1)).split()
    sys.argv = command
    slurp.prepare.prepare.main()
