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
Brings together the geometry functions using OTB features, to project images into
georeferenced geometry (with superimpose) or Shareloc, to project images into sensor geometry
"""

import time

import bindings_cpp
import numpy as np
import logging
import rasterio as rio
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from shareloc.dtm_reader import dtm_reader
from shareloc.geofunctions.localization import Localization
from shareloc.geomodels import GeoModel
from shareloc.image import Image


logger = logging.getLogger("slurp")


def get_extract_roi(file_in: str, file_ref: str) -> np.ndarray:
    """
    Extract ROI 

    :param str file_in: path to the image to crop
    :param str file_ref: path to the input reference image
    """
    start_time = time.time()

    # Open the reference image to get its bounds in its CRS
    with rasterio.open(file_ref) as ref_src:
        ref_bounds = ref_src.bounds
        ref_crs = ref_src.crs

    # Open the input image (to be cropped)
    with rasterio.open(file_in) as in_src:
        in_crs = in_src.crs

        # Transform reference bounds to input CRS if necessary
        if ref_crs != in_crs:
            ref_bounds = transform_bounds(ref_crs, in_crs, *ref_bounds)

        # Create a window in the input image matching the transformed reference bounds
        window = from_bounds(*ref_bounds, transform=in_src.transform)
        window = window.round_offsets().round_lengths()

        # Read the window as a numpy array
        roi = in_src.read(window=window)

    logger.info("Extract ROI in %.2f seconds.", time.time() - start_time)

    return roi


def compute_dtm_footprint(geom_model, sensor_image, data_img):
    """
    Compute DTM footprint from supposed sensor_image footprint 
    to load only a portion of the global DTM.

    Load Shareloc direct loc function force shareloc to use north for
    vertical direction (compatible with images with positive
    or negative y spacing.

    Args:
        geom_model (GeoModel): GeoModel factory:
        A class designed for registered all available geomodels
        and instantiate them when needed.
        sensor_image (str): path to input image in its raw geometry (sensor)
        data_img (Any): Image opened at sensor_image path.
    """
    loc_alt_min = Localization(
        geom_model,
        elevation=-2000.0,
        image=Image(sensor_image, vertical_direction="north"),
        epsg=4326,
    )
    loc_alt_max = Localization(
        geom_model,
        elevation=10000.0,
        image=Image(sensor_image, vertical_direction="north"),
        epsg=4326,
    )

    # lower right (lr) and upper left (ul) corners coordinates, with two altitudes
    lon_lat_lr_corner_altmax = loc_alt_max.direct(
        data_img.shape[0] - 1, data_img.shape[1] - 1
    )[0][0:2]
    lon_lat_ul_corner_altmax = loc_alt_max.direct(0, 0)[0][0:2]

    lon_lat_lr_corner_altmin = loc_alt_min.direct(
        data_img.shape[0] - 1, data_img.shape[1] - 1
    )[0][0:2]
    lon_lat_ul_corner_altmin = loc_alt_min.direct(0, 0)[0][0:2]

    # lon = coords[0] / lat = coords[1]
    # depending on the projection, the upper left corner has not always
    # the lowest longiture or the highest latitude !!
    min_lat = min(
        lon_lat_lr_corner_altmax[1],
        lon_lat_ul_corner_altmax[1],
        lon_lat_lr_corner_altmin[1],
        lon_lat_ul_corner_altmin[1],
    )
    min_lon = min(
        lon_lat_lr_corner_altmax[0],
        lon_lat_ul_corner_altmax[0],
        lon_lat_lr_corner_altmin[0],
        lon_lat_ul_corner_altmin[0],
    )
    max_lat = max(
        lon_lat_lr_corner_altmax[1],
        lon_lat_ul_corner_altmax[1],
        lon_lat_lr_corner_altmin[1],
        lon_lat_ul_corner_altmin[1],
    )
    max_lon = max(
        lon_lat_lr_corner_altmax[0],
        lon_lat_ul_corner_altmax[0],
        lon_lat_lr_corner_altmin[0],
        lon_lat_ul_corner_altmin[0],
    )

    # compute usable extent and add a margin
    footprint_dtm = [min_lat, min_lon, max_lat, max_lon]
    footprint_dtm[0] -= 0.5
    footprint_dtm[1] -= 0.5
    footprint_dtm[2] += 0.5
    footprint_dtm[3] += 0.5

    return footprint_dtm


def compute_interpolation_grid(sensor_image, dtm_file, geoid_file, step=30):
    """
    Compute an interpolation grid (lons/lats) -> x/y coords that will help reproject
    global georeferenced data in sensor geometry

    :param str sensor_image: path to input image in its raw geometry (sensor)
    :param str dtm_file: path to the DTM
    :param str geoid_file: path to the Geoid
    :param int step: grid step (default, 30)
    """
    # Import image geometrical model
    geom_model_optim = GeoModel(sensor_image, "RPCoptim")

    # TODO : refac :
    # New function 'compute grids' ?  to be discussed !

    # Read image and retrieve its bbox coordinates
    data_img = rio.open(sensor_image)
    nb_row, nb_col = data_img.profile["height"], data_img.profile["width"]
    transf = data_img.profile["transform"]
    pix_row, pix_col = np.abs(transf[4]), np.abs(transf[0])
    bbox = np.array(
        [
            [0, 0],
            [nb_col + step, 0],
            [nb_col + step, nb_row + step],
            [0, nb_row + step],
        ]
    )

    # Resampled image grid (same geometry as sensor_image, but every n step)
    x = np.arange(0, nb_col + step, step)
    y = np.arange(0, nb_row + step, step)
    col, row = np.meshgrid(x, y)
    grid_nb_cols, grid_nb_rows = col.shape

    cols_rows = np.vstack((col.flatten(), row.flatten()))
    cols_rows_t = cols_rows.transpose()

    # Full image grid
    all_x = np.arange(0, nb_col, pix_col)
    all_y = np.arange(0, nb_row, pix_row)
    all_col, all_row = np.meshgrid(all_x, all_y)
    all_coords = np.vstack((all_col.flatten(), all_row.flatten())).transpose()

    footprint_dtm = compute_dtm_footprint(
        geom_model_optim, sensor_image, data_img
    )

    image = Image(sensor_image, vertical_direction="north")
    dtm_image = dtm_reader(
        dtm_file,
        geoid_file,
        roi=footprint_dtm,
        roi_is_in_physical_space=True,
        fill_nodata=None,
        fill_value=0.0,
    )
    dtm_optim = bindings_cpp.DTMIntersection(
        dtm_image.epsg,
        dtm_image.alt_data,
        dtm_image.nb_rows,
        dtm_image.nb_columns,
        dtm_image.transform,
    )

    # Localization model taking into account local DTM
    loc_optim = Localization(
        geom_model_optim, elevation=dtm_optim, image=image, epsg=4326
    )

    # TODO : refac :
    # New function 'Get bbox pixel coordinates in lat/lon' ?

    coords_bbox_min = None
    coords_bbox_max = None
    alt_min = dtm_optim.get_alt_min()
    alt_max = dtm_optim.get_alt_max()

    for coord in bbox:
        # row /col
        latlon_alt_min = loc_optim.direct(
            coord[1], coord[0], h=alt_min, using_geotransform=True
        )
        latlon_alt_max = loc_optim.direct(
            coord[1], coord[0], h=alt_max, using_geotransform=True
        )
        coords_bbox_min = (
            latlon_alt_min
            if coords_bbox_min is None
            else np.append(coords_bbox_min, latlon_alt_min, axis=0)
        )
        coords_bbox_max = (
            latlon_alt_max
            if coords_bbox_max is None
            else np.append(coords_bbox_max, latlon_alt_max, axis=0)
        )
    min_lon = np.minimum(
        np.min(coords_bbox_min[:, 0]), np.min(coords_bbox_max[:, 0])
    )
    max_lon = np.maximum(
        np.max(coords_bbox_min[:, 0]), np.max(coords_bbox_max[:, 0])
    )
    min_lat = np.minimum(
        np.min(coords_bbox_min[:, 1]), np.min(coords_bbox_max[:, 1])
    )
    max_lat = np.maximum(
        np.max(coords_bbox_min[:, 1]), np.max(coords_bbox_max[:, 1])
    )

    # Direct localization of resampled image to get associated terrain coordinates
    coords_4326 = loc_optim.direct(
        cols_rows_t[:, 0], cols_rows_t[:, 1], using_geotransform=True
    )[:, :2]

    coords_lon = coords_4326[:, 0].reshape((grid_nb_cols, grid_nb_rows))
    coords_lat = coords_4326[:, 1].reshape((grid_nb_cols, grid_nb_rows))

    # interpolate positions on image coordinates
    grid_positions = (y, x)

    roi = [min_lat, min_lon, max_lat, max_lon]

    #  grid in sensor (every n step), grid in lon/lat (every n step), grid of every pixel in sensor, roi to load on external data
    return grid_positions, (coords_lon, coords_lat), all_coords, roi


def sensor_projection(
    input_data,
    sensor_image,
    projected_data,
    grid_sensor,
    grid_geo,
    all_coords,
    roi,
):
    """
    Reproject georeferenced data into sensor geometry

    :param str input_data: path to global data (ie : Pekel, WSF, etc.) to crop and reproject
    :param str sensor_image: path to input image in its raw geometry (sensor)
    :param str dtm_file: path to the DTM
    :param str geoid_file: path to the Geoid
    :param str projected_data: path to the output projected data
    """
    try:
        from shareloc.image import Image
        from shareloc.proj_utils import transform_physical_point_to_index
    except ModuleNotFoundError as e:
        raise ImportError("\n*** Shareloc is not installed ***\n") from e

    import scipy

    print(grid_geo)
    # construct all pixels positions
    interp_lon = scipy.interpolate.interpn(
        grid_sensor,
        grid_geo[0],
        all_coords,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    interp_lat = scipy.interpolate.interpn(
        grid_sensor,
        grid_geo[1],
        all_coords,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    interp_pos = np.stack((interp_lat, interp_lon)).transpose()

    # load subset of external data
    image_roi = Image(
        input_data, read_data=True, roi=roi, roi_is_in_physical_space=True
    )

    # transform positions in pekel image positions
    indexes = transform_physical_point_to_index(
        image_roi.trans_inv, interp_pos[:, 0], interp_pos[:, 1]
    )

    # Nearest Neighbor
    # TODO use cars-resample to do linear interpolation ?
    values = np.round(indexes).astype(int)
    # get data at indexes
    image_data = image_roi.data[values[0, :], values[1, :]]

    ds_sensor_image = rio.open(sensor_image)
    profile = ds_sensor_image.profile
    nb_rows = profile["height"]
    nb_columns = profile["width"]

    reshaped_image = np.reshape(image_data, (nb_rows, nb_columns))

    ext_data = rio.open(input_data)

    # force GTiff as output
    profile.update(
        {"count": 1, "dtype": ext_data.profile["dtype"], "driver": "GTiff"}
    )

    dst2 = rio.open(projected_data, "w", **profile)
    dst2.write(reshaped_image, indexes=1)
    dst2 = None
