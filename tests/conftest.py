#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 Y T[3~[D.
#
# This file is part of slurp
#
"""Fixtures definitions"""

import json
import pytest
import os

pytest.register_assert_rewrite('tests.utils')
pytest.register_assert_rewrite('tests.validation')


def pytest_collection_modifyitems(items, config):
    # add `default` marker to all unmarked items
    for item in items:
        if not any(item.iter_markers()):
            item.add_marker("default")
    # Ensure the `default` marker is always selected for
    markexpr = config.getoption("markexpr", 'False')
    if markexpr := config.getoption('markexpr', 'False'):
        config.option.markexpr = f"({markexpr})"
    else:
        config.option.markexpr = "default or computation_and_validation"
        
        
def pytest_configure(config):
    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, 'config_tests.json')) as f:
        conf = json.load(f)
        pytest.data_dir = conf["data_dir"]
        pytest.output_dir = conf["output_dir"]
        pytest.ref_dir = conf["ref_dir"]
    pytest.main_config = os.path.join(current_dir, 'main_config_tests.json')
    if not os.path.exists(pytest.output_dir):
        os.makedirs(pytest.output_dir)