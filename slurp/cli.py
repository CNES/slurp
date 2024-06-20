#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 CNES
#
# This file is part of slurp
#

"""
Console script for slurp.
"""

import argparse
import sys


def main():
    """Console script for slurp."""
    parser = argparse.ArgumentParser()
    parser.add_argument("_", nargs="*")
    args = parser.parse_args()

    print("Arguments: " + str(args._))
    print("Replace this message by putting your code into " "slurp.cli.main")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
