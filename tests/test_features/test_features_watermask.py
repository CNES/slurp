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


def write_command_compute_watermask(
    nb_workers,
    main_config,
    features_test_img,
    output_dir,
    ref_dir,
    valid_stack=None,
):
    """Builds a command string to compute a water mask using the watermask module."""
    output_image = get_output_path(
        features_test_img, "watermask", output_dir, remove=True
    )
    if valid_stack is None:
        valid_stack = get_aux_path(features_test_img, "valid_stack", ref_dir)

    return (
        f"watermask.py {main_config} "
        f"-file_vhr {features_test_img} "
        f"-n_workers {nb_workers} "
        f"-watermask {output_image} "
        f"-valid {valid_stack}"
    )


@pytest.mark.features
def test_hand_strict(main_config, features_test_img, output_dir, ref_dir):
    """Tests the water mask computation with HAND strict filtering enabled.
    hand_strict: Use not(pekelxx) for other (no water) samples."""
    command = f"{write_command_compute_watermask(1, main_config, features_test_img, output_dir, ref_dir)} -hand_strict".split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.ci
def test_hand_strict_ci(
    main_config, features_test_img, output_dir, ref_dir, valid_stack
):
    """Run test_hand_strict with a specified valid_stack (for GithubCI)."""
    command = f"{write_command_compute_watermask(1, main_config, features_test_img, output_dir, ref_dir, valid_stack)} -hand_strict".split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.features
def test_simple_ndwi_threshold(
    main_config, features_test_img, output_dir, ref_dir
):
    """Tests the water mask computation with a simple NDWI threshold enabled.
    simple_ndwi_threshold: Compute water mask as a simple NDWI threshold,
    useful in arid places where no water is known by Peckel"""
    command = f"{write_command_compute_watermask(1, main_config, features_test_img, output_dir, ref_dir)} -simple_ndwi_threshold True ".split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.ci
def test_simple_ndwi_threshold_ci(
    main_config, features_test_img, output_dir, ref_dir, valid_stack
):
    """Run test_simple_nwdi_threshold with specified valid_stack (for GithubCI)."""
    command = f"{write_command_compute_watermask(1, main_config, features_test_img, output_dir, ref_dir, valid_stack)} -simple_ndwi_threshold True ".split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.features
@pytest.mark.parametrize("samples_method", ["random", "smart", "grid"])
def test_samples_method(
    main_config, features_test_img, output_dir, ref_dir, samples_method
):
    """Tests the water mask computation with different sample selection methods.
    samples_method: Select method for choosing learning samples"""
    command = f"{write_command_compute_watermask(1, main_config, features_test_img, output_dir, ref_dir)} -samples_method {samples_method}".split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.ci
@pytest.mark.parametrize("samples_method", ["random", "smart", "grid"])
def test_samples_method_ci(
    main_config,
    features_test_img,
    output_dir,
    ref_dir,
    valid_stack,
    samples_method,
):
    """Run test_samples_method with a specified valid_stack (for GithubCI)."""
    command = (
        write_command_compute_watermask(
            1, main_config, features_test_img, output_dir, ref_dir, valid_stack
        )
        + f" -samples_method {samples_method}"
    ).split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.features
@pytest.mark.parametrize(
    "nb_samples_water,nb_samples_other", [(10000, 1000), (100, 100)]
)
def test_nb_samples(
    main_config,
    features_test_img,
    output_dir,
    ref_dir,
    nb_samples_water,
    nb_samples_other,
):
    """Tests the water mask computation with different sample counts for water and other classes.
    nb_samples_water: Number of samples in water for learning.
    nb_samples_other: Number of samples in 'other' class for learning."""
    command = (
        write_command_compute_watermask(
            1, main_config, features_test_img, output_dir, ref_dir
        )
        + f" -nb_samples_water {nb_samples_water} -nb_samples_other {nb_samples_other}"
    ).split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.features
def test_nb_samples_auto(main_config, features_test_img, output_dir, ref_dir):
    """Tests the water mask computation with automatic sample count selection."""
    command = f"{write_command_compute_watermask(1, main_config, features_test_img, output_dir, ref_dir)} -nb_samples_auto".split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.features
def test_pekel_filter(main_config, features_test_img, output_dir, ref_dir):
    """Tests the water mask computation with the Pekel filter disabled:
    Deactivate postprocess with pekel which only keeps surfaces already known by pekel.
    """
    command = f"{write_command_compute_watermask(1, main_config, features_test_img, output_dir, ref_dir)} -no_pekel_filter".split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.ci
def test_pekel_filter_ci(
    main_config, features_test_img, output_dir, ref_dir, valid_stack
):
    """Run test_pekel_filter with a specified valid_stack (for GithubCI)."""
    command = f"{write_command_compute_watermask(1, main_config, features_test_img, output_dir, ref_dir, valid_stack)} -no_pekel_filter".split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.features
def test_hand_filter(main_config, features_test_img, output_dir, ref_dir):
    """Tests the water mask computation with HAND filtering enabled:
    Postprocess with Hand (set to 0 when hand > thresh), incompatible with hand_strict
    """
    command = f"{write_command_compute_watermask(1, main_config, features_test_img, output_dir, ref_dir)} -hand_filter".split()
    sys.argv = command
    slurp.masks.watermask.main()


@pytest.mark.ci
def test_hand_filter_ci(
    main_config, features_test_img, output_dir, ref_dir, valid_stack
):
    """Run test_hand_filter with a specified valid_stack (for GithubCI)."""
    command = f"{write_command_compute_watermask(1, main_config, features_test_img, output_dir, ref_dir, valid_stack)} -hand_filter".split()
    sys.argv = command
    slurp.masks.watermask.main()
