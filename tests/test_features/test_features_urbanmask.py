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


def write_command_compute_urbanmask(
    nb_workers,
    main_config,
    features_test_img,
    output_dir,
    ref_dir,
    valid_stack=None,
):
    """Builds a command string to compute an urban mask using the urbanmask module."""
    output_image = get_output_path(
        features_test_img, "urbanmask", output_dir, remove=True
    )
    if valid_stack is None:
        valid_stack = get_aux_path(features_test_img, "valid_stack", ref_dir)

    return (
        f"urbanmask.py {main_config} "
        f"-file_vhr {features_test_img} "
        f"-n_workers {nb_workers} "
        f"-urbanmask {output_image} "
        f"-valid {valid_stack} "
    )


@pytest.mark.features
@pytest.mark.parametrize("vegmask_min_value", [0, 21, 1000])
def test_vegmask_max_value(
    vegmask_min_value, main_config, features_test_img, output_dir, ref_dir
):
    """Tests the urban mask computation with different vegetation mask minimum values.
    vegmask_min_value: Vegetation min value for vegetated areas :
    all pixels with lower value will be predicted"""
    command = (
        write_command_compute_urbanmask(
            1, main_config, features_test_img, output_dir, ref_dir
        )
        + f"-vegmask_min_value {vegmask_min_value} "
    ).split()
    sys.argv = command
    slurp.masks.urbanmask.main()


@pytest.mark.ci
@pytest.mark.parametrize("vegmask_min_value", [0, 21, 1000])
def test_vegmask_max_value_ci(
    vegmask_min_value,
    main_config,
    features_test_img,
    output_dir,
    ref_dir,
    valid_stack,
):
    """Run the test test_vegmask_max_value with a specified valid_stack (for GithubCI)."""
    command = (
        write_command_compute_urbanmask(
            1, main_config, features_test_img, output_dir, ref_dir, valid_stack
        )
        + f"-vegmask_min_value {vegmask_min_value}"
    ).split()
    sys.argv = command
    slurp.masks.urbanmask.main()


@pytest.mark.features
@pytest.mark.parametrize(
    "nb_samples_other,nb_samples_urban", [(0, 0), (5000, 1000)]
)
def test_nb_samples(
    nb_samples_other,
    nb_samples_urban,
    main_config,
    features_test_img,
    output_dir,
    ref_dir,
):
    """Tests the urban mask computation with different sample counts for other and urban classes.
    nb_samples_other: Number of samples in other for learning.
    nb_samples_urban: Number of samples in buildings for learning"""
    command = (
        write_command_compute_urbanmask(
            1, main_config, features_test_img, output_dir, ref_dir
        )
        + f"-nb_samples_other {nb_samples_other} -nb_samples_urban {nb_samples_urban}"
    ).split()
    sys.argv = command
    slurp.masks.urbanmask.main()


@pytest.mark.ci
@pytest.mark.parametrize(
    "nb_samples_other,nb_samples_urban", [(0, 0), (5000, 1000)]
)
def test_nb_samples_ci(
    nb_samples_other,
    nb_samples_urban,
    main_config,
    features_test_img,
    output_dir,
    ref_dir,
    valid_stack,
):
    """Run test_nb_samples with a specified valid_stack (for GithubCI)."""
    command = (
        write_command_compute_urbanmask(
            1, main_config, features_test_img, output_dir, ref_dir, valid_stack
        )
        + f"-nb_samples_other {nb_samples_other} -nb_samples_urban {nb_samples_urban}"
    ).split()
    sys.argv = command
    slurp.masks.urbanmask.main()


@pytest.mark.features
@pytest.mark.parametrize(
    "main_config,features_test_img,ref_dir,valid_stack,aux_layer",
    [
        (
            "/work/scratch/data/amselln/uc_toulouse_with_MNH/MNH_config.json",
            "/work/scratch/data/amselln/uc_toulouse_with_MNH/xt_Toulouse_with_MNH.tif",
            "/work/scratch/data/amselln/uc_toulouse_with_MNH/ref",
            "/work/scratch/data/amselln/uc_toulouse_with_MNH/out/valid_stack.tif",
            "/work/scratch/data/amselln/uc_toulouse_with_MNH/xt_MNH.tif",
        ),
    ],
)
def test_layers_aux_file(
    main_config,
    features_test_img,
    output_dir,
    ref_dir,
    valid_stack,
    aux_layer,
):
    """Tests the urban mask computation with auxiliary raster layers
    to ensure the -layers argument correctly ingests external files."""

    command = (
        write_command_compute_urbanmask(
            1,
            main_config,
            features_test_img,
            output_dir,
            ref_dir,
            valid_stack,
        )
        + f"-layers {aux_layer} "
    ).split()

    sys.argv = command
    slurp.masks.urbanmask.main()