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

"""Brings together some useful functions"""

import os
import time
import json
from os import makedirs, path, remove
from importlib.resources import files

import logging.config
import numpy as np
import psutil

from slurp.tools import io_utils
from slurp.tools.constant import NODATA_INT8

logger = logging.getLogger("slurp")

def store_arglist(parser):
    """
    Stores the list of argument names from the CLI parser into a JSON file.
    This file is then used to overwrite the main configuration file parameters.
    """
    arglist = []
    for arg in parser._actions:
        if arg.dest not in ["help"]:
            arglist.append(arg.dest)
    with open("args_list.json", 'w') as f:
        json.dump(arglist, f)

def setup_logging(config_file : str):
    with open(config_file) as f_in:
        config = json.load(f_in)

    logging.config.dictConfig(config)


def parse_args(keys, logs_to_file, main_config):
    '''
    Parse command line arguments.
    Setup logging with a configuration file based on logs_to_file option value.
    '''
    argsdict = io_utils.read_json(
        main_config, keys)

    # Read the list back from the JSON file
    with open("args_list.json", 'r') as f:
        cli_params = json.load(f)
    remove("args_list.json")

    if logs_to_file:
        config_file = files("slurp.tools.logs").joinpath("out2json.json")
        if not path.exists("logs"):
            makedirs("logs")
    else:
        config_file = files("slurp.tools.logs").joinpath("out2stdout.json")
    setup_logging(config_file)
    return argsdict, cli_params


def convert_time(seconds):
    full_time = time.gmtime(seconds)
    return time.strftime("%H:%M:%S", full_time)


def compute_mask(im_ref: np.ndarray, thresh_ref: list) -> list:
    """
    Compute mask with one or multiple threshold values

    :param np.ndarray im_ref: input image
    :param list thresh_ref: list of threshold values
    :returns: list of masks for each threshold value
    """
    mask_ref = np.zeros(im_ref.shape)
    for thresh in thresh_ref:
        local_mask = im_ref > thresh
        mask_ref = np.where(local_mask, local_mask, mask_ref)

    return mask_ref


def compute_mask_threshold(
    input_buffers: list, input_profiles: list, params: dict
) -> np.ndarray:
    """
    Compute boolean mask with threshold value

    :param list input_buffers: Input image and valid stack [input_image, valid_stack]
    :param list input_profiles: image profile (not used but necessary for eoscale)
    :param dict params: dictionary of arguments, must contain the key "threshold"
    :returns: computed mask
    """
    mask = np.where(input_buffers[0][0] > params["threshold"], 1, 0)
    mask = np.where(input_buffers[1][0] != 1, NODATA_INT8, mask)

    return mask


def display_mem_usage(debug_mode, message):
    """If we are in debug mode, the memory usage is displayed."""
    if debug_mode:
        pid = os.getpid()
        python_process = psutil.Process(pid)
        memory_use = (
            python_process.memory_info()[0] / 2.0**30
        )  # memory use in GB...I think
        logger.debug(f">> {message} >> Mem usage : {memory_use} Gb")
