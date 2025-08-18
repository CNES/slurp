#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022-2024 CNES
#
# This file is part of slurp
#
"""Test urban mask with differents features and different arguments values"""

import sys

import pytest

import slurp.masks.urbanmask
from tests.utils import get_aux_path, get_output_path


def write_command_compute_urbanmask(nb_workers, valid_stack=None):
    output_image = get_output_path(
        pytest.features_test_img, "urbanmask", remove=True
    )
    if valid_stack is None:
        valid_stack = get_aux_path(pytest.features_test_img, "valid_stack")

    return f"urbanmask.py {pytest.main_config} -file_vhr {pytest.features_test_img} -n_workers {nb_workers} -urbanmask {output_image} -valid {valid_stack} "


@pytest.mark.fast
@pytest.mark.parametrize("vegmask_min_value", [0, 21, 1000])
def test_vegmask_max_value(vegmask_min_value):
    command = (
        write_command_compute_urbanmask(1)
        + f"-vegmask_min_value {vegmask_min_value}"
    ).split()
    sys.argv = command
    slurp.masks.urbanmask.main()


@pytest.mark.fast
@pytest.mark.parametrize(
    "nb_samples_other,nb_samples_urban", [(0, 0), (5000, 1000)]
)
def test_nb_samples(nb_samples_other, nb_samples_urban):
    command = (
        write_command_compute_urbanmask(1)
        + f"-nb_samples_other {nb_samples_other} -nb_samples_other {nb_samples_other}"
    ).split()
    sys.argv = command
    slurp.masks.urbanmask.main()


@pytest.mark.fast
def test_files_layers():
    command = (
        write_command_compute_urbanmask(1)
        + "-layers ['/work/datalake/static_aux/MASQUES/WSF/WSF2019_v1/WSF2019_v1.vrt']"
    ).split()
    sys.argv = command
    slurp.masks.urbanmask.main()
