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

"""Test the transformations dispatched by apply_morpho"""

import numpy as np
import pytest

from slurp.post_process.morphology import apply_morpho

BINARY_KEYS = (
    "binary_closing",
    "binary_dilation",
    "binary_erosion",
    "binary_opening",
)


def build_square():
    """Build a 7x7 mask holding a single 3x3 square.

    :return: boolean mask
    """
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True
    return mask


def build_square_with_an_isolated_pixel():
    """Build a 9x9 mask holding a 4x4 square and a one pixel object.

    :return: boolean mask
    """
    mask = np.zeros((9, 9), dtype=bool)
    mask[1, 1] = True
    mask[4:8, 4:8] = True
    return mask


def build_square_with_a_one_pixel_hole():
    """Build a 9x9 mask holding a 5x5 square dug with a one pixel hole.

    :return: boolean mask
    """
    mask = np.zeros((9, 9), dtype=bool)
    mask[2:7, 2:7] = True
    mask[4, 4] = False
    return mask


@pytest.mark.ci
def test_binary_erosion_keeps_only_the_center_of_a_square():
    """A radius one erosion of a 3x3 square leaves its center pixel."""
    expected = np.zeros((7, 7), dtype=bool)
    expected[3, 3] = True

    eroded = apply_morpho(build_square(), "binary_erosion", 1)

    assert np.array_equal(eroded, expected)


@pytest.mark.ci
def test_binary_dilation_grows_a_square():
    """A radius one dilation adds the four neighbours of every pixel."""
    dilated = apply_morpho(build_square(), "binary_dilation", 1)

    assert dilated.sum() == 21


@pytest.mark.ci
def test_binary_opening_removes_an_isolated_pixel():
    """Opening erodes then dilates, so a lone pixel does not survive."""
    mask = build_square_with_an_isolated_pixel()

    opened = apply_morpho(mask, "binary_opening", 1)

    assert not opened[1, 1]


@pytest.mark.ci
def test_binary_closing_fills_a_one_pixel_hole():
    """Closing dilates then erodes, so a lone hole gets filled."""
    mask = build_square_with_a_one_pixel_hole()

    closed = apply_morpho(mask, "binary_closing", 1)

    assert closed[4, 4]
    assert closed.sum() == mask.sum() + 1


@pytest.mark.ci
def test_remove_small_objects_drops_objects_of_the_given_size():
    """An object holding as many pixels as the threshold is removed."""
    mask = build_square_with_an_isolated_pixel()

    cleaned = apply_morpho(mask, "remove_small_objects", 1)

    assert not cleaned[1, 1]
    assert cleaned.sum() == 16


@pytest.mark.ci
def test_remove_small_objects_keeps_objects_above_the_given_size():
    """An object larger than the threshold is left untouched."""
    mask = build_square_with_an_isolated_pixel()

    cleaned = apply_morpho(mask, "remove_small_objects", 15)

    assert cleaned.sum() == 16


@pytest.mark.ci
def test_remove_small_holes_fills_holes_of_the_given_size():
    """A hole holding as many pixels as the threshold is filled."""
    mask = build_square_with_a_one_pixel_hole()

    filled = apply_morpho(mask, "remove_small_holes", 1)

    assert filled[4, 4]
    assert filled.sum() == 25


@pytest.mark.parametrize("key", BINARY_KEYS)
@pytest.mark.ci
def test_binary_transformations_return_a_boolean_array(key):
    """Binary transformations answer booleans whatever the input type."""
    mask = build_square().astype(np.uint8)

    transformed = apply_morpho(mask, key, 1)

    assert transformed.dtype == np.bool_


@pytest.mark.ci
def test_an_unknown_key_is_refused():
    """A key without transformation raises instead of answering None."""
    with pytest.raises(NotImplementedError):
        apply_morpho(build_square(), "unknown_key", 1)
