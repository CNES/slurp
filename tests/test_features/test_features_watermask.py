#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022-2024 CNES
#
# This file is part of slurp
#
"""Test water mask with differents features and different arguments values"""

import sys

import pytest

import slurp.masks.watermask
from tests.utils import get_aux_path, get_output_path


def write_command_compute_watermask(nb_workers, valid_stack=None):
    """Builds a command string to compute a water mask using the watermask module."""
    output_image = get_output_path(
        pytest.features_test_img, "watermask", remove=True
    )
    if valid_stack is None:
        valid_stack = get_aux_path(pytest.features_test_img, "valid_stack")

    return f"watermask.py {pytest.main_config} -file_vhr {pytest.features_test_img} -n_workers {nb_workers} -watermask {output_image} -valid {valid_stack} "

@pytest.mark.features
def test_hand_strict():
    """Tests the water mask computation with HAND strict filtering enabled.
    hand_strict: Use not(pekelxx) for other (no water) samples."""
    command = f"{write_command_compute_watermask(1)}-hand_strict".split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.ci
def test_hand_strict_ci():
    """Run test_hand_strict with a specified valid_stack (for GithubCI)."""
    command = f"{write_command_compute_watermask(1, pytest.valid_stack)}-hand_strict".split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.features
def test_simple_ndwi_threshold():
    """Tests the water mask computation with a simple NDWI threshold enabled.
    simple_ndwi_threshold: Compute water mask as a simple NDWI threshold,
    useful in arid places where no water is known by Peckel"""
    command = f"{write_command_compute_watermask(1)}-simple_ndwi_threshold True ".split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.ci
def test_simple_ndwi_threshold_ci():
    """Run test_simple_nwdi_threshold with specified valid_stack (for GithubCI)."""
    command = f"{write_command_compute_watermask(1, pytest.valid_stack)}-simple_ndwi_threshold True ".split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.features
@pytest.mark.parametrize("samples_method", ["random", "smart", "grid"])
def test_samples_method(samples_method):
    """Tests the water mask computation with different sample selection methods.
    samples_method: Select method for choosing learning samples"""
    command = f"{write_command_compute_watermask(1)}-samples_method {samples_method}".split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.ci
@pytest.mark.parametrize("samples_method", ["random", "smart", "grid"])
def test_samples_method_ci(samples_method):
    """Run test_samples_method with a specified valid_stack (for GithubCI)."""
    command = (
        write_command_compute_watermask(1, pytest.valid_stack)
        + f"-samples_method {samples_method}"
    ).split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.features
@pytest.mark.parametrize(
    "nb_samples_water,nb_samples_other", [(10000, 1000), (100, 100)]
)
def test_nb_samples(nb_samples_water, nb_samples_other):
    """Tests the water mask computation with different sample counts for water and other classes.
    nb_samples_water: Number of samples in water for learning.
    nb_samples_other: Number of samples in 'other' class for learning."""
    command = (
        write_command_compute_watermask(1)
        + f"-nb_samples_water {nb_samples_water} -nb_samples_other {nb_samples_other}"
    ).split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.features
def test_nb_samples_auto():
    """Tests the water mask computation with automatic sample count selection."""
    command = f"{write_command_compute_watermask(1)}-nb_samples_auto".split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.features
def test_pekel_filter():
    """Tests the water mask computation with the Pekel filter disabled:
    Deactivate postprocess with pekel which only keeps surfaces already known by pekel.
    """
    command = f"{write_command_compute_watermask(1)}-no_pekel_filter".split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.ci
def test_pekel_filter_ci():
    """Run test_pekel_filter with a specified valid_stack (for GithubCI)."""
    command = f"{write_command_compute_watermask(1, pytest.valid_stack)}-no_pekel_filter".split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.features
def test_hand_filter():
    """Tests the water mask computation with HAND filtering enabled:
    Postprocess with Hand (set to 0 when hand > thresh), incompatible with hand_strict
    """
    command = f"{write_command_compute_watermask(1)}-hand_filter".split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.ci
def test_hand_filter_ci():
    """Run test_hand_filter with a specified valid_stack (for GithubCI)."""
    command = f"{write_command_compute_watermask(1, pytest.valid_stack)}-hand_filter".split()
    sys.argv = command
    slurp.masks.watermask.main()
