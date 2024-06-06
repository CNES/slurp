#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 Y T[3~[D.
#
# This file is part of slum
#
"""Definition of validation functions"""

import os
import rasterio as rio
import pytest
import numpy as np


def compare_datasets(ds_new_mask, ds_ref_mask):
    assert ds_new_mask.profile == ds_ref_mask.profile
    assert ds_new_mask.tags() == ds_ref_mask.tags()
    assert ds_new_mask.bounds == ds_ref_mask.bounds
    assert ds_new_mask.colorinterp == ds_ref_mask.colorinterp
    assert ds_new_mask.tag_namespaces() == ds_ref_mask.tag_namespaces()


def validate_mask(new_file, key, valid_pixels=True):
    filename = os.path.basename(new_file)
    ref_file = os.path.join(pytest.ref_dir, key, "ref_" + filename)
    assert os.path.exists(ref_file)
    
    ds_ref_mask = rio.open(ref_file)
    ds_new_mask = rio.open(new_file)
    
    # Dataset comparison
    compare_datasets(ds_new_mask, ds_ref_mask)
    
    # Pixels comparison
    if valid_pixels:
        assert np.array_equal(ds_new_mask.read(1), ds_ref_mask.read(1))
            
    ds_new_mask.close()
    ds_ref_mask.close()

