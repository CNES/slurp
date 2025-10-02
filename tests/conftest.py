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

"""Fixtures definitions"""

import json
import os

import pytest

pytest.register_assert_rewrite("tests.utils")
pytest.register_assert_rewrite("tests.validation")


def pytest_collection_modifyitems(items, config):
    # add `default` marker to all unmarked items
    for item in items:
        if not any(item.iter_markers()):
            item.add_marker("default")
    # Ensure the `default` marker is always selected for
    markexpr = config.getoption("markexpr", "False")
    if markexpr := config.getoption("markexpr", "False"):
        config.option.markexpr = f"({markexpr})"
    else:
        config.option.markexpr = "default or computation_and_validation"


def pytest_addoption(parser):
    parser.addoption("--config", action="store", default="config_tests.json")
    parser.addoption(
        "--main_config", action="store", default="main_config_tests.json"
    )


def pytest_configure(config):
    config_file = config.getoption("config")
    main_config = config.getoption("main_config")
    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, config_file)) as f:
        conf = json.load(f)
        pytest.data_dir = conf["data_dir"]
        pytest.sensor_geom_dir = conf["sensor_geom_dir"]
        pytest.features_test_img = conf["features_test_img"]
        pytest.output_dir = os.path.join(current_dir, conf["output_dir"])
        pytest.ref_dir = conf["ref_dir"]
        pytest.pekel = conf["pekel"]
        pytest.hand = conf["hand"]
        pytest.wsf = conf["wsf"]
        pytest.valid_stack = conf["valid_stack"]
    pytest.main_config = os.path.join(current_dir, main_config)
    if not os.path.exists(pytest.output_dir):
        os.makedirs(pytest.output_dir)
