#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 Y T[3~[D.
#
# This file is part of slum
#
"""Tests for `slum` package."""

# Third party imports
import pytest

# slum imports
import slum


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # Example to edit
    return "response"


def test_content(response):  # pylint: disable=redefined-outer-name
    """Sample pytest test function with the pytest fixture as an argument."""
    # Example to edit
    print(response)


def test_slum():
    """Sample pytest slum module test function"""
    assert slum.__author__ == "Y T[3~[D"
    assert slum.__email__ == "yannick.tanguy@cnes.fr"
