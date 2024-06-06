#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 Y T[3~[D.
#
# This file is part of slum
#
"""Definition of global functions."""

import os
import glob
import json
import pytest
import contextlib


def get_files_to_process(key):
    all_input_folder = os.path.join(pytest.data_dir, "all")
    key_input_folder = os.path.join(pytest.data_dir, key)
    return glob.glob(all_input_folder + "/*.tif") + glob.glob(key_input_folder + "/*.tif")


def get_output_path(file, key):
    assert os.path.exists(file)
    assert os.path.exists(pytest.output_dir)
    filename = os.path.basename(file)
    output_image = os.path.join(pytest.output_dir, key + "_" + filename)
    return output_image


def remove_file(file):
    with contextlib.suppress(FileNotFoundError):
        os.remove(file)
        