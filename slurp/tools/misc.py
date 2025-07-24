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


"""Brings together miscellaneous display functions"""
import linecache
import os
import logging
import tracemalloc

import psutil

logger = logging.getLogger("slurp")

def display_top(snapshot, key_type="lineno", limit=10):
    """Print a snapshot of momentary used memory"""

    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        )
    )
    top_stats = snapshot.statistics(key_type)

    logger.info(f"Top {limit} lines")
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        logger.info(
            f"#{index}: {filename}:{frame.lineno}: {stat.size / 1024:.1f} KiB"
        )
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            logger.info(f"    {line}")

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        logger.info(f"{len(other)} other: {size / 1024:.1f} KiB")
    total = sum(stat.size for stat in top_stats)
    logger.info(f"Total allocated size: {total / 1024:.1f} KiB")


def display_mem(step):
    mem_used = psutil.Process().memory_info().rss / (1024 * 1024)
    logger.info(">>>" + str(step) + "\t >>> Mem used : \t" + str(mem_used) + " Mb")
