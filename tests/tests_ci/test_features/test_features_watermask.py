#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022-2024 CNES
#
# This file is part of slurp
#
"""Test water mask with differents features and different arguments values"""

import os
import subprocess
import sys

import pytest

import slurp.masks.watermask
from tests.utils import get_aux_path, get_output_path


def write_command_compute_watermask(nb_workers, valid_stack=None):
    output_image = get_output_path(
        pytest.features_test_img, "watermask", remove=True
    )

    return f"watermask.py {pytest.main_config} -file_vhr {pytest.features_test_img} -n_workers {nb_workers} -watermask {output_image} -valid {pytest.valid_stack} "


#@pytest.mark.fast
#def test_files_layers():
#    command = (
#        write_command_compute_watermask(1)
#        + f"-layers ['/work/datalake/static_aux/MASQUES/WSF/WSF2019_v1/WSF2019_v1.vrt']"
#    ).split()
#    sys.argv = command
#    slurp.masks.watermask.main()


@pytest.mark.fast
def test_hand_strict():
    command = (write_command_compute_watermask(1) + f"-hand_strict").split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.fast
def test_simple_ndwi_threshold():
    command = (
        write_command_compute_watermask(1) + f"-simple_ndwi_threshold True "
    ).split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.fast
@pytest.mark.parametrize("samples_method", ["random", "smart", "grid"])
def test_samples_method(samples_method):
    command = (
        write_command_compute_watermask(1) + f"-samples_method {samples_method}"
    ).split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.fast
@pytest.mark.parametrize(
    "nb_samples_water,nb_samples_other", [(10000, 1000), (0, 0)]
)
def test_nb_samples(nb_samples_water, nb_samples_other):
    command = (
        write_command_compute_watermask(1)
        + f"-nb_samples_water {nb_samples_water} -nb_samples_other {nb_samples_other}"
    ).split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.fast
def test_nb_samples_auto():
    command = (write_command_compute_watermask(1) + f"-nb_samples_auto").split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.fast
def test_pekel_filter():
    command = (write_command_compute_watermask(1) + f"-no_pekel_filter").split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.fast
def test_hand_filter():
    command = (write_command_compute_watermask(1) + f"-hand_filter").split()
    sys.argv = command
    slurp.masks.watermask.main()
