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

"""  Brings together functions used by eoscale"""
import copy
import numpy as np


def print_dataset_infos(name, profile, prefix=""):
    """Print information about rasterio dataset."""

    print()
    print(prefix, "Image name :", name)
    print(prefix, "Image size :", profile["width"], "x", profile["height"])
    print(prefix, "Image bands :", profile["count"])
    print(prefix, "Image types :", profile["dtype"])
    print(prefix, "Image nodata :", profile["nodata"])
    print(prefix, "Image crs :", profile["crs"])
    print(prefix, "Image transform :", profile["transform"])
    print(prefix, "Image driver :", profile["driver"])
    print()


def concatenate_samples(output_scalars, chunk_output_scalars, tile):
    output_scalars.append(chunk_output_scalars[0])


# Profiles

def single_float_profile(input_profiles: list, map_params):
    """ Define profile for eoscale """
    profile = input_profiles[0]
    profile["count"] = 1
    profile["dtype"] = np.float32
    profile["compress"] = "deflate"
    profile["driver"] = "GTiff"

    return profile


def single_bool_profile(input_profiles: list, map_params):
    """ Define profile for eoscale """
    profile = input_profiles[0]
    profile["count"] = 1
    profile["dtype"] = bool
    profile["compress"] = "deflate"
    profile["driver"] = "GTiff"
    
    return profile
    

def single_uint8_1b_profile(input_profiles: list, map_params):
    """ Define profile for eoscale """
    profile = input_profiles[0]
    profile["count"] = 1
    profile["dtype"] = np.uint8
    profile["nbits"] = 1
    profile["compress"] = "deflate"
    profile["nodata"] = None
    profile["driver"] = "GTiff"

    return profile


def single_uint8_profile(input_profiles: list, map_params):
    """ Define profile for eoscale """
    profile = input_profiles[0]
    profile["count"] = 1
    profile["dtype"] = np.uint8
    profile["compress"] = "deflate"
    profile["nodata"] = 255
    profile["driver"] = "GTiff"
    
    return profile


def single_int16_profile(input_profiles: list, map_params):
    """ Define profile for eoscale """
    profile = input_profiles[0]
    profile["count"] = 1
    profile["dtype"] = np.int16
    profile["nodata"] = 32767
    profile["compress"] = "deflate"
    profile["driver"] = "GTiff"

    return profile


def single_uint16_profile(input_profiles: list, map_params):
    """ Define profile for eoscale """
    profile = input_profiles[0]
    profile["count"] = 1
    profile["dtype"] = np.uint16
    profile["nodata"] = 32767
    profile["compress"] = "deflate"
    profile["driver"] = "GTiff"
    
    return profile


def single_int32_profile(input_profiles: list, map_params):
    """ Define profile for eoscale """
    profile = input_profiles[0]
    profile["count"] = 1
    profile["dtype"] = np.int32
    profile["compress"] = "deflate"
    profile["driver"] = "GTiff"

    return profile


def three_uint8_profile(input_profiles: list, map_params):
    """ Define profiles for eoscale """
    profile = input_profiles[0]
    profile["count"] = 3
    profile["dtype"] = np.uint8
    profile["compress"] = "deflate"
    profile["nodata"] = 255
    profile["driver"] = "GTiff"
    
    return profile


def double_int_profile(input_profiles: list, map_params):
    """ Define profiles for eoscale """
    profile1 = input_profiles[0]
    profile1["count"] = 1
    profile1["dtype"] = np.uint8
    profile1["nodata"] = 255
    profile1["compress"] = "deflate"
    profile1["driver"] = "GTiff"
    
    # avoid to modify profile1
    profile2 = copy.deepcopy(profile1)
       
    return [profile1, profile2] 
