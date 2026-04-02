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
This module is used to run multiprocessing in slurp
"""

import math
from multiprocessing.synchronize import Lock
from typing import Callable, List, Tuple, Union, Optional

import numpy as np
import tqdm

from slurp.eomultiprocessing.slurp_manager import slurpContextManager
from slurp.eomultiprocessing.utils import MpTile, read_window, write_window


def compute_mp_strips(
    image_height: int,
    image_width: int,
    nb_workers: int,
    stable_margin: int,
    along_line: bool = False,
    specific_strip_size: Union[int, None] = None,
) -> List[MpTile]:
    """
    Compute processing strips for multiprocessing based on image geometry.

    The image is divided into strips either horizontally or vertically
    depending on the value of ``along_line``. Each strip corresponds to
    a processing tile assigned to a worker. Optional margins are added
    to ensure stable border processing when neighborhood operations are
    required.

    Args:
        image_height (int): Height of the input image.
        image_width (int): Width of the input image.
        nb_workers (int): Number of workers used for multiprocessing.
        stable_margin (int): Margin size added around each strip to
            guarantee stable computation near tile borders.
        along_line (bool): If True, generate vertical strips
            (split along X direction). If False, generate horizontal
            strips (split along Y direction).
        specific_strip_size (Union[int, None]): Optional fixed strip size.
            If None, the strip size is automatically computed as
            image_height or image_width divided by the number of workers.

    Returns:
        List[MpTile]: List of MpTile objects describing the strips,
        including their coordinates, dimensions, and margins.
    """
    total_size = image_width if along_line else image_height
    if specific_strip_size:
        strip_size = specific_strip_size
    else:
        strip_size = total_size // nb_workers

    nb_tiles = total_size // strip_size
    if total_size % strip_size > 0:
        nb_tiles += 1

    strips = []

    for t in range(nb_tiles):
        start = t * strip_size
        end = min((t + 1) * strip_size - 1, total_size - 1)
        start_margin = stable_margin if start - stable_margin >= 0 else start
        end_margin: int = stable_margin if end + stable_margin <= total_size - 1 else total_size - 1 - end

        if along_line:
            strips.append(
                MpTile(
                    start_x=start,
                    start_y=0,
                    end_x=end,
                    end_y=image_height - 1,
                    top_margin=0,
                    right_margin=end_margin,
                    bottom_margin=0,
                    left_margin=start_margin,
                    height=image_height,
                    width=end - start + 1,
                    height_margin=image_height,
                    width_margin=(end + end_margin) - (start - start_margin) + 1,
                )
            )
        else:
            strips.append(
                MpTile(
                    start_x=0,
                    start_y=start,
                    end_x=image_width - 1,
                    end_y=end,
                    top_margin=start_margin,
                    right_margin=0,
                    bottom_margin=end_margin,
                    left_margin=0,
                    height=end - start + 1,
                    width=image_width,
                    height_margin=(end + end_margin) - (start - start_margin) + 1,
                    width_margin=image_width,
                )
            )

    return strips


def compute_mp_tiles(
    image_height: int,
    image_width: int,
    nb_workers: int,
    stable_margin: int,
    tile_mode: bool,
    specific_tile_size: Union[int, None] = None,
    strip_along_lines: bool = False,
    tile_max_size: int = 0
) -> List[MpTile]:
    """
    Compute multiprocessing tiles or strips based on the input image geometry.

    This function determines how an image should be split for parallel
    processing. Depending on ``tile_mode``, the image is either divided
    into square tiles or into horizontal/vertical strips. Each resulting
    region is represented as an ``MpTile`` object containing coordinates
    and margins used for stable border computations.

    Args:
        image_height (int): Height of the input image.
        image_width (int): Width of the input image.
        nb_workers (int): Number of workers used for multiprocessing.
        stable_margin (int): Margin added around each tile or strip to
            ensure correct processing at boundaries.
        tile_mode (bool): If True, the image is divided into square tiles.
            If False, the image is divided into strips.
        specific_tile_size (Union[int, None]): Optional fixed tile or strip
            size. If None, the size is automatically computed based on the
            image size and the number of workers.
        strip_along_lines (bool): If using strips (tile_mode=False),
            controls the orientation of the strips:
            True for horizontal strips, False for vertical strips.

    Returns:
        List[MpTile]: List of MpTile objects describing each tile or strip,
        including coordinates, dimensions, and margins required for
        processing.
    """

    if tile_mode:

        nb_pixels_per_worker: int = (image_width * image_height) // nb_workers

        if specific_tile_size:
            # User override always wins
            tile_size = specific_tile_size

        else:
            auto_tile_size = int(math.sqrt(nb_pixels_per_worker))

            if tile_max_size > 0:
                # Memory protection
                tile_size = min(auto_tile_size, tile_max_size)
            else:
                tile_size = auto_tile_size

        nb_tiles_x = image_width // tile_size
        nb_tiles_y = image_height // tile_size

        if image_width % tile_size > 0:
            nb_tiles_x += 1
        if image_height % tile_size > 0:
            nb_tiles_y += 1

        strips: list = []

        for ty in range(nb_tiles_y):

            for tx in range(nb_tiles_x):

                # Determine the stable and unstable boundaries of the tile
                start_x = tx * tile_size
                start_y = ty * tile_size
                end_x = min((tx + 1) * tile_size - 1, image_width - 1)
                end_y = min((ty + 1) * tile_size - 1, image_height - 1)

                top_margin = stable_margin if start_y - stable_margin >= 0 else start_y
                left_margin = stable_margin if start_x - stable_margin >= 0 else start_x
                bottom_margin = stable_margin if end_y + stable_margin <= image_height - 1 else image_height - 1 - end_y
                right_margin = stable_margin if end_x + stable_margin <= image_width - 1 else image_width - 1 - end_x

                strips.append(
                    MpTile(
                        start_x=start_x,
                        start_y=start_y,
                        end_x=end_x,
                        end_y=end_y,
                        top_margin=top_margin,
                        right_margin=right_margin,
                        bottom_margin=bottom_margin,
                        left_margin=left_margin,
                        height=end_y - start_y + 1,
                        width=end_x - start_x + 1,
                        height_margin=(end_y + bottom_margin) - (start_y - top_margin) + 1,
                        width_margin=(end_x + right_margin) - (start_x - left_margin) + 1,
                    )
                )

    else:
        strips = compute_mp_strips(
            image_height=image_height,
            image_width=image_width,
            nb_workers=nb_workers,
            stable_margin=stable_margin,
            along_line=strip_along_lines,
            specific_strip_size=specific_tile_size,
        )

    return strips

def mp_n_to_m_images(
    inputs: list,
    image_height: int,
    image_width: int,
    output_keys: List[str],
    output_profiles: List[dict],
    context_manager,
    func: Callable,
    func_parameters: Union[dict, None] = None,
    aux_inputs: Optional[List[np.ndarray]] = None,
    stable_margin: int = 0,
    tile_mode: Union[bool, None] = None,
    specific_tile_size: Union[int, None] = None,
    strip_along_lines: bool = False,
    binary: bool = False,
    debug: bool = False,
) -> Union[List[str], List[np.ndarray], Tuple[np.ndarray]]:
    """
    Process n input images to produce m outputs using optional multiprocessing.
    """

    func_parameters, aux_inputs, has_aux = _validate_inputs(
        inputs,
        func,
        context_manager,
        func_parameters,
        aux_inputs,
        image_height,
        image_width,
    )

    # --------------------------------------------------------
    # SINGLE PROCESS MODE
    # --------------------------------------------------------
    if context_manager.pool is None:
        return _run_singleprocess(
            inputs,
            func,
            func_parameters,
            aux_inputs,
            has_aux,
            context_manager,
            output_keys,
            output_profiles,
            binary,
            debug,
        )

    # --------------------------------------------------------
    # MULTIPROCESSING
    # --------------------------------------------------------
    tiles = _compute_tiles(
        image_height,
        image_width,
        stable_margin,
        context_manager,
        tile_mode,
        specific_tile_size,
        strip_along_lines,
    )

    if context_manager.in_memory:
        return _run_in_memory(
            inputs,
            aux_inputs,
            has_aux,
            tiles,
            func,
            func_parameters,
            context_manager,
            image_height,
            image_width,
            output_profiles,
            output_keys,
            binary,
            debug,
        )

    return _run_streaming(
        inputs,
        aux_inputs,
        has_aux,
        tiles,
        func,
        func_parameters,
        context_manager,
        output_profiles,
        output_keys,
        binary,
        debug,
    )

def _find_spatial_axes(arr: np.ndarray, h: int, w: int):
    """
    Find which axes correspond to spatial dimensions (H, W)
    regardless of axis order.

    Returns:
        (h_axis, w_axis)
    """

    matches = []

    for i, size_i in enumerate(arr.shape):
        for j, size_j in enumerate(arr.shape):
            if i == j:
                continue

            if size_i == h and size_j == w:
                matches.append((i, j))

    if not matches:
        raise ValueError(
            f"Cannot find spatial axes matching ({h},{w}) "
            f"in array shape {arr.shape}"
        )

    # take first valid match
    return matches[0]
# ============================================================
# VALIDATION
# ============================================================

def _validate_inputs(
    inputs,
    func,
    context_manager,
    func_parameters,
    aux_inputs,
    h,
    w,
):
    if len(inputs) < 1:
        raise ValueError("At least one input image must be given.")

    if func is None:
        raise ValueError("A filter must be set!")

    if context_manager is None:
        raise ValueError("The Context Manager must be given!")

    func_parameters = func_parameters or {}

    has_aux = aux_inputs is not None and len(aux_inputs) > 0
    aux_inputs = aux_inputs or []

    for arr in aux_inputs:
        if not _has_spatial_dims(arr, h, w):
            raise ValueError(
                f"Aux input shape {arr.shape} "
                f"does not contain spatial dims ({h},{w})"
            )

    return func_parameters, aux_inputs, has_aux


# ============================================================
# HELPERS
# ============================================================

def _build_params(base_params, aux, has_aux):
    params = dict(base_params)
    if has_aux:
        params["aux_inputs"] = aux
    return params

def _has_spatial_dims(arr, h, w):
    try:
        _find_spatial_axes(arr, h, w)
        return True
    except ValueError:
        return False

def _compute_tiles(
    image_height,
    image_width,
    stable_margin,
    context_manager,
    tile_mode,
    specific_tile_size,
    strip_along_lines,
):
    return compute_mp_tiles(
        image_height=image_height,
        image_width=image_width,
        stable_margin=stable_margin,
        nb_workers=context_manager.nb_workers,
        tile_mode=tile_mode
        if tile_mode is not None
        else context_manager.tile_mode,
        specific_tile_size=specific_tile_size,
        strip_along_lines=strip_along_lines,
        tile_max_size=context_manager.tile_max_size
    )


# ============================================================
# SINGLE PROCESS
# ============================================================

def _run_singleprocess(
    inputs,
    func,
    func_parameters,
    aux_inputs,
    has_aux,
    context_manager,
    output_keys,
    output_profiles,
    binary,
    debug,
):
    for inp in inputs:
        if isinstance(inp, str):
            raise ValueError(
                "Without multiprocessing the inputs must be numpy arrays."
            )

    params = _build_params(func_parameters, aux_inputs, has_aux)
    output = func(*inputs, **params)

    outputs = _normalize_output(output)

    if debug and context_manager.dev_mode:
        _write_debug_outputs(
            outputs,
            context_manager,
            output_keys,
            output_profiles,
            binary,
        )

    return outputs


# ============================================================
# IN MEMORY MULTIPROCESSING
# ============================================================

def _run_in_memory(
    inputs,
    aux_inputs,
    has_aux,
    tiles,
    func,
    func_parameters,
    context_manager,
    image_height,
    image_width,
    output_profiles,
    output_keys,
    binary,
    debug,
):

    jobs = []

    for tile in tiles:
        tile_inputs = [_slice_array(arr, tile, image_height, image_width) for arr in inputs]

        tile_aux = None
        if has_aux:
            tile_aux = [_slice_array(arr, tile, image_height, image_width) for arr in aux_inputs]

        params = _build_params(func_parameters, tile_aux, has_aux)

        jobs.append((tile_inputs, func, params, tile))

    out_chunks = context_manager.pool.starmap(
        mp_execute_from_arrays,
        tqdm.tqdm(jobs, total=len(jobs)),
    )

    outputs = _reconstruct_outputs(
        out_chunks,
        image_height,
        image_width,
        output_profiles,
    )

    if debug and context_manager.dev_mode:
        _write_debug_outputs(
            outputs,
            context_manager,
            output_keys,
            output_profiles,
            binary,
        )

    return outputs


# ============================================================
# STREAMING MODE
# ============================================================

def _run_streaming(
    inputs,
    aux_inputs,
    has_aux,
    tiles,
    func,
    func_parameters,
    context_manager,
    output_profiles,
    output_keys,
    binary,
    debug,
):

    key = (
        context_manager.debug_key
        if debug and context_manager.dev_mode
        else context_manager.tmp_key
    )

    output_paths = [
        context_manager.get_path(k, key) for k in output_keys
    ]

    params = _build_params(
        func_parameters,
        aux_inputs if has_aux else None,
        has_aux,
    )

    jobs = [
        (
            inputs,
            output_paths,
            func,
            params,
            output_profiles,
            context_manager.lock,
            tile,
            binary,
        )
        for tile in tiles
    ]

    context_manager.pool.starmap(
        mp_execute_from_paths,
        tqdm.tqdm(jobs, total=len(jobs)),
    )

    return output_paths


# ============================================================
# UTILITIES
# ============================================================

def _slice_array(arr, tile, h, w):

    h_axis, w_axis = _find_spatial_axes(arr, h, w)

    slices = [slice(None)] * arr.ndim

    slices[h_axis] = slice(
        tile.start_y - tile.top_margin,
        tile.end_y + tile.bottom_margin + 1,
    )

    slices[w_axis] = slice(
        tile.start_x - tile.left_margin,
        tile.end_x + tile.right_margin + 1,
    )

    return arr[tuple(slices)]


def _normalize_output(output):
    if isinstance(output, np.ndarray):
        return [output]

    if isinstance(output, tuple):
        return list(output)

    raise ValueError(f"Wrong output type {type(output)}.")


def _reconstruct_outputs(chunks, h, w, output_profiles):
    """
    Reconstruct outputs from spatial chunks.

    Works with chunked datasets where worker outputs
    are tile-sized instead of full-image sized.
    """

    outputs = []
    first = chunks[0]["data"]

    spatial_axes_cache = []

    # --------------------------------------------------
    # Detect spatial axes from FIRST CHUNK
    # --------------------------------------------------
    first_tile = chunks[0]["tile"]

    for i, arr in enumerate(first):

        if arr.ndim < 2:
            raise RuntimeError(
                f"Invalid output dimension {arr.shape}"
            )

        # spatial dims = dims matching tile size
        candidate_axes = []

        for ax, size in enumerate(arr.shape):
            if size in (
                first_tile.end_y - first_tile.start_y + 1,
                first_tile.end_x - first_tile.start_x + 1,
            ):
                candidate_axes.append(ax)

        if len(candidate_axes) < 2:
            raise RuntimeError(
                f"Cannot infer spatial axes from chunk shape {arr.shape}"
            )

        h_axis, w_axis = candidate_axes[:2]
        spatial_axes_cache.append((h_axis, w_axis))

        # allocate full output
        shape = list(arr.shape)
        shape[h_axis] = h
        shape[w_axis] = w

        outputs.append(
            np.zeros(
                tuple(shape),
                dtype=output_profiles[i]["dtype"],
            )
        )

    # --------------------------------------------------
    # Reconstruct tiles
    # --------------------------------------------------
    for chunk in chunks:
        tile = chunk["tile"]

        for i, out in enumerate(outputs):

            data = chunk["data"][i]
            h_axis, w_axis = spatial_axes_cache[i]

            hh = data.shape[h_axis]
            ww = data.shape[w_axis]

            slices = [slice(None)] * out.ndim

            slices[h_axis] = slice(
                tile.start_y,
                tile.start_y + hh,
            )

            slices[w_axis] = slice(
                tile.start_x,
                tile.start_x + ww,
            )

            out[tuple(slices)] = data

    return outputs


def _write_debug_outputs(
    outputs,
    context_manager,
    output_keys,
    output_profiles,
    binary,
):
    for i, data in enumerate(outputs):
        context_manager.write_tif(
            data,
            output_keys[i],
            output_profiles[i],
            binary,
        )


def mp_execute_from_paths(
    input_paths: List[str],
    output_paths: List[str],
    func: Callable,
    func_parameters: dict,
    output_profiles: List[dict],
    lock: Lock,
    tile: MpTile,
    binary: bool,
) -> None:
    """
    Execute a function on raster windows in multiprocessing and save the outputs.

    This method is designed to run within a multiprocessing pool. Each process
    reads the portion of each input raster corresponding to the provided tile,
    executes the given function on this window, and writes the results to disk.
    A lock is used to prevent concurrent writes to the same output files.

    Args:
        input_paths (List[str]): List of input raster file paths.
        output_paths (List[str]): List of output raster file paths.
        func (Callable): Processing function applied to each tile.
        func_parameters (dict): Dictionary of parameters to pass to `func`.
        output_profiles (List[dict]): Rasterio profiles for each output image.
        lock (Lock): Multiprocessing lock to avoid simultaneous writes.
        tile (MpTile): Tile defining the window to process (coordinates, size, margins).
        binary (bool): If True, outputs are written as binary rasters.

    Returns:
        None

    Notes:
        - The function slices inputs and outputs using the tile's margins to ensure
          stable border processing.
        - Supports multiple outputs (tuple of arrays) or single output arrays.
    """
    # Read the inputs
    inputs = [read_window(path, tile) for path in input_paths]

    # Run the function
    outputs = func(*inputs, **func_parameters)

    # Write the outputs
    if len(output_paths) > 1:
        for index, out_path in enumerate(output_paths):
            with lock:
                write_window(
                    outputs[index][
                        tile.top_margin : tile.top_margin + tile.height,
                        tile.left_margin : tile.left_margin + tile.width,
                    ],
                    out_path,
                    output_profiles[index],
                    tile,
                    binary,
                )
    else:
        with lock:
            write_window(
                outputs[
                    tile.top_margin : tile.top_margin + tile.height,
                    tile.left_margin : tile.left_margin + tile.width,
                ],
                output_paths[0],
                output_profiles[0],
                tile,
                binary,
            )


def mp_execute_from_arrays(
    inputs: List[np.ndarray],
    func: Callable,
    func_parameters: dict,
    tile: MpTile,
) -> dict:
    """
    Execute a function on numpy array tiles in multiprocessing and return results.

    This method is designed to run within a multiprocessing pool. Each process
    slices the given input arrays according to the tile's coordinates and margins,
    applies the function, and returns a dictionary containing the results and
    tile information. Margins are removed from the returned data.

    Args:
        inputs (List[np.ndarray]): List of input arrays corresponding to the tile.
        func (Callable): Processing function applied to each tile.
        func_parameters (dict): Dictionary of parameters to pass to `func`.
        tile (MpTile): Tile defining the window to process (coordinates, size, margins).

    Returns:
        dict: Dictionary with keys:
            - "data": List of numpy arrays (one per output) cropped to the tile
              without margins.
            - "tile": MpTile object corresponding to the processed tile.

    Notes:
        - Supports functions returning a single array or a tuple of arrays.
        - Useful for in-memory multiprocessing workflows.
    """
    outputs = func(*inputs, **func_parameters)
    if not isinstance(outputs, tuple):
        return {
            "data": [
                outputs[
                    tile.top_margin : tile.top_margin + tile.height,
                    tile.left_margin : tile.left_margin + tile.width,
                ]
            ],
            "tile": tile,
        }

    return {
        "data": [
            output[
                tile.top_margin : tile.top_margin + tile.height,
                tile.left_margin : tile.left_margin + tile.width,
            ]
            for output in outputs
        ],
        "tile": tile,
    }

def mp_n_to_m_scalars(
    inputs: list,
    image_height: int,
    image_width: int,
    context_manager: slurpContextManager,
    func: Callable,
    reducer: Callable,
    func_parameters: Union[dict, None] = None,
    aux_inputs: Optional[List[np.ndarray]] = None,
    stable_margin: int = 0,
    tile_mode: Union[bool, None] = None,
    specific_tile_size: Union[int, None] = None,
    strip_along_lines: bool = False,
) -> Union[float, Tuple]:
    """
    Process n input images to produce m scalar outputs using a parallel
    map/reduce-like paradigm with optional auxiliary inputs.

    This function slices input images (and optional auxiliary images)
    into tiles or strips depending on ``tile_mode`` and dispatches them
    to a processing function ``func``. The results of each tile are
    then aggregated using the provided ``reducer`` function.

    Args:
        inputs (list): List of input images as numpy arrays (all must
            have the same geometry: image_height x image_width).
        image_height (int): Height of the input images.
        image_width (int): Width of the input images.
        context_manager (slurpContextManager): Context manager handling
            multiprocessing pool, I/O, and locking.
        func (Callable): Function to process each tile. Must return a
            scalar or tuple of scalars for each tile.
        reducer (Callable): Function to aggregate tile-level outputs
            into global scalar(s).
        func_parameters (Union[dict, None]): Optional dictionary of
            parameters passed to ``func``.
        aux_inputs (Optional[List[np.ndarray]]): Optional list of
            auxiliary images of same dimensions as inputs. These are
            sliced similarly to main inputs and passed to ``func``.
        stable_margin (int): Margin added around tiles/strips to ensure
            stable border computations.
        tile_mode (Union[bool, None]): If True, split images into square
            tiles; if False, split images into strips. If None, uses
            default from context_manager.
        specific_tile_size (Union[int, None]): Optional fixed tile/strip
            size. If None, size is computed automatically.
        strip_along_lines (bool): If using strips (tile_mode=False),
            True for horizontal strips, False for vertical strips.

    Returns:
        Union[float, Tuple]: Aggregated scalar result(s) computed from
        all tiles. Can be a single scalar or a tuple of scalars depending
        on the processing function ``func`` and the ``reducer`` used.

    Raises:
        ValueError: If inputs or aux_inputs are missing or have incompatible
        dimensions, or if required functions (func/reducer) or context_manager
        are not provided.

    Notes:
        - All input images and aux_inputs must have identical dimensions.
        - If context_manager.pool is None, the function runs in serial.
        - In multiprocessing mode, tiles/strips are dispatched to workers
          and the results are merged by the reducer.
    """

    func_parameters, aux_inputs, has_aux = _validate_inputs(
        inputs,
        func,
        context_manager,
        func_parameters,
        aux_inputs,
        image_height,
        image_width,
    )

    if reducer is None:
        raise ValueError("Reducer must be provided.")

    # ------------------------
    # SINGLE PROCESS
    # ------------------------
    if context_manager.pool is None:
        return _run_scalar_singleprocess(
            inputs,
            func,
            reducer,
            func_parameters,
            aux_inputs,
            has_aux,
        )

    # ------------------------
    # MULTIPROCESS
    # ------------------------
    tiles = _compute_tiles(
        image_height,
        image_width,
        stable_margin,
        context_manager,
        tile_mode,
        specific_tile_size,
        strip_along_lines,
    )

    if context_manager.in_memory:
        results = _run_scalar_in_memory(
            inputs,
            aux_inputs,
            has_aux,
            tiles,
            func,
            func_parameters,
            context_manager,
            image_height,
            image_width,
        )
    else:
        results = _run_scalar_streaming(
            inputs,
            aux_inputs,
            has_aux,
            tiles,
            func,
            func_parameters,
            context_manager,
        )

    return reducer(results)
def mp_execute_scalar_from_arrays(
    inputs: List[np.ndarray],
    func: Callable,
    func_parameters: dict,
    tile: MpTile,
):
    outputs = func(*inputs, **func_parameters)
    return outputs

def mp_execute_scalar_from_paths(
    input_paths: List[str],
    func: Callable,
    func_parameters: dict,
    tile: MpTile,
):
    inputs = [read_window(path, tile) for path in input_paths]
    outputs = func(*inputs, **func_parameters)
    return outputs

def _run_scalar_singleprocess(
    inputs,
    func,
    reducer,
    func_parameters,
    aux_inputs,
    has_aux,
):
    params = _build_params(func_parameters, aux_inputs, has_aux)

    result = func(*inputs, **params)

    return reducer([result])

def _run_scalar_in_memory(
    inputs,
    aux_inputs,
    has_aux,
    tiles,
    func,
    func_parameters,
    context_manager,
    h,
    w,
):

    jobs = []

    for tile in tiles:

        tile_inputs = [
            _slice_array(arr, tile, h, w)
            for arr in inputs
        ]

        tile_aux = None
        if has_aux:
            tile_aux = [
                _slice_array(arr, tile, h, w)
                for arr in aux_inputs
            ]

        params = _build_params(func_parameters, tile_aux, has_aux)

        jobs.append(
            (tile_inputs, func, params, tile)
        )

    return context_manager.pool.starmap(
        mp_execute_scalar_from_arrays,
        tqdm.tqdm(jobs, total=len(jobs)),
    )

def _run_scalar_streaming(
    inputs,
    aux_inputs,
    has_aux,
    tiles,
    func,
    func_parameters,
    context_manager,
):

    params = _build_params(
        func_parameters,
        aux_inputs if has_aux else None,
        has_aux,
    )

    jobs = [
        (inputs, func, params, tile)
        for tile in tiles
    ]

    return context_manager.pool.starmap(
        mp_execute_scalar_from_paths,
        tqdm.tqdm(jobs, total=len(jobs)),
    )
# ============================================================
# MAP-REDUCE LABEL REINDEXING
# ============================================================

def _apply_global_label_mapping(chunks):
    """
    Ensure global uniqueness of labels across tiles.

    Each tile segmentation is independently indexed.
    This function applies a cumulative offset so that
    labels become globally unique before reconstruction.
    """

    global_offset = 0

    for chunk in chunks:

        data_list = chunk["data"]

        new_data_list = []

        for arr in data_list:

            if not np.issubdtype(arr.dtype, np.integer):
                new_data_list.append(arr)
                continue

            mask = arr > 0

            if np.any(mask):
                arr = arr.copy()
                arr[mask] += global_offset
                global_offset = arr.max()

            new_data_list.append(arr)

        chunk["data"] = new_data_list

    return chunks


# ============================================================
# IN MEMORY MULTIPROCESSING WITH MAP-REDUCE
# ============================================================

def _run_in_memory_with_mapping(
    inputs,
    aux_inputs,
    has_aux,
    tiles,
    func,
    func_parameters,
    context_manager,
    image_height,
    image_width,
    output_profiles,
    output_keys,
    binary,
    debug,
):

    jobs = []

    for tile in tiles:

        tile_inputs = [
            _slice_array(arr, tile, image_height, image_width)
            for arr in inputs
        ]

        tile_aux = None
        if has_aux:
            tile_aux = [
                _slice_array(arr, tile, image_height, image_width)
                for arr in aux_inputs
            ]

        params = _build_params(func_parameters, tile_aux, has_aux)

        jobs.append((tile_inputs, func, params, tile))

    # ---------------- MAP ----------------
    out_chunks = context_manager.pool.starmap(
        mp_execute_from_arrays,
        tqdm.tqdm(jobs, total=len(jobs)),
    )

    # ---------------- REDUCE ----------------
    out_chunks = _apply_global_label_mapping(out_chunks)

    # ---------------- MERGE ----------------
    outputs = _reconstruct_outputs(
        out_chunks,
        image_height,
        image_width,
        output_profiles,
    )

    if debug and context_manager.dev_mode:
        _write_debug_outputs(
            outputs,
            context_manager,
            output_keys,
            output_profiles,
            binary,
        )

    return outputs


def mp_n_to_m_images_with_mapping(
    inputs: list,
    image_height: int,
    image_width: int,
    output_keys: List[str],
    output_profiles: List[dict],
    context_manager,
    func: Callable,
    func_parameters: Union[dict, None] = None,
    aux_inputs: Optional[List[np.ndarray]] = None,
    stable_margin: int = 0,
    tile_mode: Union[bool, None] = None,
    specific_tile_size: Union[int, None] = None,
    strip_along_lines: bool = False,
    binary: bool = False,
    debug: bool = False,
):
    """
    Same as mp_n_to_m_images but applies a Map-Reduce
    global mapping step before reconstruction.

    Designed for algorithms producing local labels
    (segmentation, clustering, connected components).
    """

    func_parameters, aux_inputs, has_aux = _validate_inputs(
        inputs,
        func,
        context_manager,
        func_parameters,
        aux_inputs,
        image_height,
        image_width,
    )

    if context_manager.pool is None:
        return _run_singleprocess(
            inputs,
            func,
            func_parameters,
            aux_inputs,
            has_aux,
            context_manager,
            output_keys,
            output_profiles,
            binary,
            debug,
        )

    tiles = _compute_tiles(
        image_height,
        image_width,
        stable_margin,
        context_manager,
        tile_mode,
        specific_tile_size,
        strip_along_lines,
    )

    if context_manager.in_memory:
        return _run_in_memory_with_mapping(
            inputs,
            aux_inputs,
            has_aux,
            tiles,
            func,
            func_parameters,
            context_manager,
            image_height,
            image_width,
            output_profiles,
            output_keys,
            binary,
            debug,
        )

    raise RuntimeError(
        "Mapping mode currently supported only in-memory."
    )


