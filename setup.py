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

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

extensions = [
    Extension("stats", ["slum/stats/cysrc/stats.pyx"])
]

compiler_directives = { "language_level": 3, "embedsignature": True}
extensions = cythonize(extensions, compiler_directives=compiler_directives)


try:
    setup(
        ext_modules=extensions,
        packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
    )
except Exception:
    print(
        "\n\nAn error occurred while building the project, "
        "please ensure you have the most updated version of pip, setuptools, "
        "setuptools_scm and wheel with:\n"
        "   pip install -U pip setuptools setuptools_scm wheel\n\n"
    )
    raise
