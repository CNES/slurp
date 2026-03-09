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

"""Brings together functions used by eoscale"""
import copy
import logging

import numpy as np

from slurp.tools.constant import COMPRESSION, DRIVER, NODATA_INT8, NODATA_INT16

logger = logging.getLogger("slurp")



# Profiles
def single_float_profile(input_profiles: list):
    """Define profile for eoscale"""
    profile = input_profiles[0]
    profile["count"] = 1
    profile["dtype"] = np.float32
    profile["compress"] = "deflate"
    profile["driver"] = "GTiff"

    return profile


def single_bool_profile(input_profiles: list):
    """Define profile for eoscale"""
    profile = input_profiles[0]
    profile["count"] = 1
    profile["dtype"] = bool
    profile["compress"] = COMPRESSION.lower()
    profile["driver"] = DRIVER

    return profile


def single_uint8_1b_profile(input_profiles: list):
    """Define profile for eoscale"""
    profile = input_profiles[0]
    profile["count"] = 1
    profile["dtype"] = np.uint8
    profile["nbits"] = 1
    profile["compress"] = COMPRESSION.lower()
    profile["nodata"] = None
    profile["driver"] = DRIVER

    return profile


def single_uint8_profile(input_profiles: list):
    """Define profile for eoscale"""
    profile = input_profiles[0]
    profile["count"] = 1
    profile["dtype"] = np.uint8
    profile["compress"] = COMPRESSION.lower()
    profile["nodata"] = NODATA_INT8
    profile["driver"] = DRIVER

    return profile


def single_int16_profile(input_profiles: list):
    """Define profile for eoscale"""
    profile = input_profiles[0]
    profile["count"] = 1
    profile["dtype"] = np.int16
    profile["nodata"] = NODATA_INT16
    profile["compress"] = COMPRESSION.lower()
    profile["driver"] = DRIVER

    return profile


def single_uint16_profile(input_profiles: list):
    """Define profile for eoscale"""
    profile = input_profiles[0]
    profile["count"] = 1
    profile["dtype"] = np.uint16
    profile["nodata"] = NODATA_INT16
    profile["compress"] = COMPRESSION.lower()
    profile["driver"] = DRIVER

    return profile


def single_int32_profile(input_profiles: list):
    """Define profile for eoscale"""
    profile = input_profiles[0]
    profile["count"] = 1
    profile["dtype"] = np.int32
    profile["compress"] = "deflate"
    profile["driver"] = "GTiff"

    return profile


def three_uint8_profile(input_profiles: list):
    """Define profiles for eoscale"""
    profile1 = input_profiles[0]
    profile1["count"] = 1
    profile1["dtype"] = np.uint8
    profile1["compress"] = COMPRESSION.lower()
    profile1["nodata"] = NODATA_INT8
    profile1["driver"] = DRIVER

    # avoid to modify profile1
    profile2 = copy.deepcopy(profile1)

    # avoid to modify profile1
    profile3 = copy.deepcopy(profile1)

    return [profile1, profile2, profile3]


def double_uint8_profile(input_profiles: list):
    """Define profiles for eoscale"""
    profile1 = input_profiles[0]
    profile1["count"] = 1
    profile1["dtype"] = np.uint8
    profile1["nodata"] = NODATA_INT8
    profile1["compress"] = COMPRESSION.lower()
    profile1["driver"] = DRIVER

    # avoid to modify profile1
    profile2 = copy.deepcopy(profile1)

    return [profile1, profile2]
