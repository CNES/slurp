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
    SLURP Context Manager to manage multiprocessing execution and safe writing of rasters.

    Responsibilities:
        - Manage a multiprocessing pool for parallel execution
        - Manage shared locks for safe writing in multiprocessing mode
        - Provide a thread-safe `write_tif` method

    Notes:
        - Does NOT handle output directory management.
        - Determines whether processing is in-memory or streaming based on params.
    """

    def __init__(self, params: dict, tile_mode: bool = False, tile_max_size: int = 0):
        """
        Initialize the SLURP context manager.

        Args:
            params (dict): Dictionary containing execution parameters:
                - "nb_max_workers" (int): maximum number of parallel workers
                - "developer_mode" (bool): if True, enables debug writing
                - "method" (str): "mem" for in-memory processing, else streaming
                - "mp_context" (str, optional): multiprocessing start method
            tile_mode (bool): Whether to split images into tiles (True) or strips (False)
            tile_max_size : int, optional
                If fixed, the maximum tile size won't be bigger than this limit (default is 0). 
                This can help to limit memory usage, when a lot of memory is allocated in the multiprocessed function
        """
        self.nb_workers: int = extract_param(params, "nb_max_workers")
        self.dev_mode: bool = extract_param(params, "developer_mode")
        self.in_memory: bool = extract_param(params, "method") == "mem"
        self.context: Optional[str] = extract_param(params, "mp_context")
        self.tile_mode: bool = tile_mode
        self.tile_max_size = tile_max_size

        self.pool: Optional[mp.pool.Pool] = None
        self.lock: Optional[mp.synchronize.Lock] = None
        self._manager: Optional[mp.Manager] = None

    def __enter__(self):  # type: ignore
        """
        Enter the context manager.

        Sets up the multiprocessing pool if nb_workers > 1.
        Initializes a lock for thread-safe writing if not in-memory.

        Returns:
            slurpContextManager: self
        """
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

            if not self.in_memory:
                self._manager = mp.Manager()
                self.lock = self._manager.Lock()
        else:
            self.in_memory = True

        return self

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore
        """
        Exit the context manager.

        Closes and joins the multiprocessing pool if used,
        and shuts down the manager if one was created.
        """
        if self.nb_workers > 1 and self.pool is not None:
            self.pool.close()
            self.pool.join()

            if not self.in_memory and self._manager is not None:
                self._manager.shutdown()

    def write_tif(
        self,
        data: np.ndarray,
        path: str,
        target_profile: Dict[str, Any],
        binary: bool = False,
    ) -> str:
        """
        Write a GeoTIFF file in a thread-safe manner.

        Handles:
            - Relative and absolute paths
            - Automatic creation of parent directories
            - Thread-safe writing when using multiprocessing

        Args:
            data (np.ndarray): Array to write to disk
            path (str): File path (relative or absolute)
            target_profile (Dict[str, Any]): Rasterio profile for the output
            binary (bool): If True, writes as binary mask

        Returns:
            str: Absolute path of the written file
        """
        full_path = Path(path).expanduser().resolve()
        full_path.parent.mkdir(parents=True, exist_ok=True)

        if self.lock is not None:
            with self.lock:
                write(data, str(full_path), target_profile, binary)
        else:
            write(data, str(full_path), target_profile, binary)

        return str(full_path)