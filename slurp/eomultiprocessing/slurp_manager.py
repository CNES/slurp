#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2022-2026 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of slurp
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
This module is used to manage the execution context of slurp.
It handles multiprocessing logic and provides a safe write_tif method.
"""

import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from slurp.eomultiprocessing.utils import write


def extract_param(params: dict, key: str) -> Any:
    if key not in params:
        raise ValueError(f"Input parameters must contain the key '{key}'")
    return params[key]


class slurpContextManager:
    """
    slurp Context Manager to manage multiprocessing execution.

    Responsibilities:
    - Manage multiprocessing pool
    - Manage shared lock (if needed)
    - Provide safe write_tif utility

    Does NOT manage output directories anymore.
    """

    def __init__(self, params: dict, tile_mode: bool = False):
        self.nb_workers: int = extract_param(params, "nb_max_workers")
        self.dev_mode: bool = extract_param(params, "developer_mode")
        self.in_memory: bool = extract_param(params, "method") == "mem"
        self.context: Optional[str] = extract_param(params, "mp_context")
        self.tile_mode: bool = tile_mode

        self.pool: Optional[mp.pool.Pool] = None
        self.lock: Optional[mp.synchronize.Lock] = None
        self._manager: Optional[mp.Manager] = None

    # ---------------------------------------------------------------------
    # Context Manager
    # ---------------------------------------------------------------------

    def __enter__(self):  # type: ignore
        if self.nb_workers > 1:

            if self.context is None:
                self.context = mp.get_start_method()

            if self.context not in mp.get_all_start_methods():
                raise ValueError(
                    f"The multiprocessing context '{self.context}' "
                    f"is not supported by your OS. "
                    f"Please choose one among {mp.get_all_start_methods()}"
                )

            ctx = mp.get_context(self.context)
            self.pool = ctx.Pool(processes=self.nb_workers)

            # Lock only needed if writing to disk with multiprocessing
            if not self.in_memory:
                self._manager = mp.Manager()
                self.lock = self._manager.Lock()

        else:
            # No multiprocessing ? always in memory
            self.in_memory = True

        return self

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore
        if self.nb_workers > 1 and self.pool is not None:
            self.pool.close()
            self.pool.join()

            if not self.in_memory and self._manager is not None:
                self._manager.shutdown()

    # ---------------------------------------------------------------------
    # IO Utilities
    # ---------------------------------------------------------------------

    def write_tif(
        self,
        data: np.ndarray,
        path: str,
        target_profile: Dict[str, Any],
        binary: bool = False,
    ) -> str:
        """
        Write a GeoTIFF file.

        Handles:
        - Relative paths
        - Absolute paths
        - Automatic directory creation
        - Thread-safe writing (if multiprocessing)

        :param data: numpy array to write
        :param path: relative or absolute file path
        :param target_profile: raster profile
        :param binary: write as binary mask if True
        :return: absolute path of written file
        """

        # Resolve path properly (handles relative, absolute, ~, etc.)
        full_path = Path(path).expanduser().resolve()

        # Ensure parent directory exists
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # If multiprocessing with disk writing ? protect write
        if self.lock is not None:
            with self.lock:
                write(data, str(full_path), target_profile, binary)
        else:
            write(data, str(full_path), target_profile, binary)

        return str(full_path)