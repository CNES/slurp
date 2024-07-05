#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 Y T[3~[D.
#
# This file is part of slurp
#
"""Definition of global functions."""

import os
import glob
import pytest
import contextlib


def get_files_to_process(key):
    all_input_folder = os.path.join(pytest.data_dir, "all")
    key_input_folder = os.path.join(pytest.data_dir, key)
    return glob.glob(all_input_folder + "/*.tif") + glob.glob(key_input_folder + "/*.tif")


def get_output_path(file, key, remove=False):
    assert os.path.exists(file), f"The file {file} doesn't exist"
    assert os.path.exists(pytest.output_dir), f"The file {pytest.output_dir} doesn't exist"
    filename = os.path.basename(file)
    output_image = os.path.join(pytest.output_dir, key + "_" + filename)    
    if remove:
        remove_file(output_image)
    return output_image


def get_aux_path(file, key):
    filename = os.path.basename(file)
    aux_image = os.path.join(pytest.ref_dir, "Prepare",  "ref_" + key + "_" + filename)
    return aux_image


def remove_file(file):
    with contextlib.suppress(FileNotFoundError):
        os.remove(file)
        