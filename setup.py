#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 CNES.
#
# This file is part of slum
#

"""
Packaging setup.py for compatibility
All packaging in setup.cfg
"""

from setuptools import setup

try:
    setup()
except Exception:
    print(
        "\n\nAn error occurred while building the project, "
        "please ensure you have the most updated version of pip, setuptools, "
        "setuptools_scm and wheel with:\n"
        "   pip install -U pip setuptools setuptools_scm wheel\n\n"
    )
    raise
