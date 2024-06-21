#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 Y T[3~[D.
#
# This file is part of slurp
#
"""Definition of validation functions"""

import os
import rasterio as rio
import pytest
import numpy as np


def get_assert_message(predict_value, ref_value, key):
    first_line = "Differences for " + key
    second_line = f"Predict {key} : {predict_value}"
    third_line = f"Ref {key} : {ref_value}"
    return first_line + '\n' + second_line + '\n' + third_line


def compare_datasets(ds_new_mask, ds_ref_mask):
    assert ds_new_mask.profile == ds_ref_mask.profile, get_assert_message(ds_new_mask.profile, ds_ref_mask.profile, "profile")
    assert ds_new_mask.tags() == ds_ref_mask.tags(), get_assert_message(ds_new_mask.tags(), ds_ref_mask.tags(), "tags")
    assert ds_new_mask.bounds == ds_ref_mask.bounds, get_assert_message(ds_new_mask.bounds, ds_ref_mask.bounds, "bounds")
    assert ds_new_mask.colorinterp == ds_ref_mask.colorinterp, get_assert_message(ds_new_mask.colorinterp, ds_ref_mask.colorinterp, "colorinterp")
    assert ds_new_mask.tag_namespaces() == ds_ref_mask.tag_namespaces(), get_assert_message(ds_new_mask.tag_namespaces(), ds_ref_mask.tag_namespaces(), "namespaces")


def validate_mask(new_file, key, valid_pixels=True):
    filename = os.path.basename(new_file)
    ref_file = os.path.join(pytest.ref_dir, key, "ref_" + filename)
    assert os.path.exists(ref_file), f"The file {ref_file} doesn't exist"
    
    ds_ref_mask = rio.open(ref_file)
    ds_new_mask = rio.open(new_file)
    
    # Dataset comparison
    compare_datasets(ds_new_mask, ds_ref_mask)
    
    # Pixels comparison
    if valid_pixels:
        nb_pix_different = np.sum(ds_new_mask.read(1) != ds_ref_mask.read(1))
        nb_pix_total = ds_ref_mask.shape[0]*ds_ref_mask.shape[1]
        #assert np.array_equal(ds_new_mask.read(1), ds_ref_mask.read(1)), f"{np.sum(ds_new_mask.read(1) != ds_ref_mask.read(1))} pixels are different"
        assert nb_pix_different/nb_pix_total < 0.2, f"{nb_pix_different} pixels are different (> 20%) : {100*nb_pix_different/nb_pix_total} %"
            
    ds_new_mask.close()
    ds_ref_mask.close()

