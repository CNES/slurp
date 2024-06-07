#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 CNES
#
# This file is part of slum
#

"""
Top-level package for slurp.
"""

from importlib.metadata import version

# version through setuptools_scm when python3 > 3.8
try:
    __version__ = version("slurp")
except Exception:  # pylint: disable=broad-except
    __version__ = "unknown"

__author__ = "CNES - Yannick TANGUY"
__email__ = "yannick.tanguy@cnes.fr"

