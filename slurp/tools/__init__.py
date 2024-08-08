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


"""
Top-level package for common tools.
"""

from importlib.metadata import version

# version through setuptools_scm when python3 > 3.8
try:
    __version__ = version("slurp")
except Exception:  # pylint: disable=broad-except
    __version__ = "unknown"

__author__ = "CNES - Yannick TANGUY"
__email__ = "yannick.tanguy@cnes.fr"
