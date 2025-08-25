#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022-2024 CNES
#
# This file is part of slurp
#
"""Test vegetation mask with differents features and different arguments values"""

import os
import subprocess
import sys

import pytest

import slurp.masks.vegetationmask
from tests.utils import get_aux_path, get_output_path


def write_command_compute_vegetationmask(nb_workers, valid_stack=None):
    output_image = get_output_path(
        pytest.features_test_img, "vegetationmask", remove=True
    )

    return f"vegetationmask.py {pytest.main_config} -file_vhr {pytest.features_test_img} -n_workers {nb_workers} -vegetationmask {output_image} -valid {pytest.valid_stack} "

@pytest.mark.fast
def test_vegetation_mask():
    command = write_command_compute_vegetationmask(1).split()
    sys.argv = command
    slurp.masks.vegetationmask.main()

