#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Brings together miscellaneous display functions """
import os
import tracemalloc
import linecache

import psutil


def display_top(snapshot, key_type="lineno", limit=10):
    """ Print a snapshot of momentary used memory """
    
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print(f"Top {limit} lines")
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print(f"#{index}: {filename}:{frame.lineno}: {stat.size / 1024:.1f} KiB")
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print(f"    {line}")

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print(f"{len(other)} other: {size / 1024:.1f} KiB")
    total = sum(stat.size for stat in top_stats)
    print(f"Total allocated size: {total / 1024:.1f} KiB")


def display_mem(step):
    mem_used = psutil.Process().memory_info().rss / (1024 * 1024)
    print(">>>"+str(step)+"\t >>> Mem used : \t"+str(mem_used)+" Mb")
