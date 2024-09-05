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
georeferenced geometry (with superimpose) or Sharloc, to project images into sensor geometry
"""

import time
import rasterio as rio
import numpy as np

def superimpose(file_in: str, file_ref: str, file_out: str):
    import otbApplication as otb
    """
    Superimpose using OTB

    :param str file_in: path to the image to reproject into the geometry of the reference input
    :param str file_ref: path to the input reference image
    :param str file_out: path for the output reprojected image
    :param type_out: OTB type for the output image
    """
    ds = rio.open(file_in)
    # default value
    output_dtype = otb.ImagePixelType_float
    if ds.profile['dtype'] == "uint8":
        output_dtype = otb.ImagePixelType_uint8
    elif ds.profile['dtype'] == "int16":
        output_dtype = otb.ImagePixelType_int16
    elif ds.profile['dtype'] == "uint16":
        output_dtype = otb.ImagePixelType_uint16
    ds = None
        
    start_time = time.time()
    app = otb.Registry.CreateApplication("Superimpose")
    app.SetParameterString("inm", file_in)
    app.SetParameterString("inr", file_ref)
    app.SetParameterString("interpolator", "nn")
    app.SetParameterString("out", file_out + "?&writerpctags=true&gdal:co:COMPRESS=DEFLATE")
    app.SetParameterOutputImagePixelType("out", output_dtype)
    app.ExecuteAndWriteOutput()

    print("Superimpose in", time.time() - start_time, "seconds.")


def rasterization(file_in: str, file_ref: str, file_out: str):
    import otbApplication as otb
    """
    Rasterization using OTB

    :param str file_in: path to the image to rasterize
    :param str file_ref: path to the input reference image
    :param str file_out: path for the output reprojected image
    :param type_out: OTB type for the output image
    """
    ds = rio.open(file_in)
    # default value
    output_dtype = otb.ImagePixelType_float
    if ds.profile['dtype'] == "uint8":
        output_dtype = otb.ImagePixelType_uint8
    elif ds.profile['dtype'] == "int16":
        output_dtype = otb.ImagePixelType_int16
    elif ds.profile['dtype'] == "uint16":
        output_dtype = otb.ImagePixelType_uint16
    ds = None
    
    start_time = time.time()
    app = otb.Registry.CreateApplication("Rasterization")
    app.SetParameterString("in", file_in)
    app.SetParameterString("im", file_ref)
    app.SetParameterFloat("background", 0)
    app.SetParameterString("mode", "binary")
    app.SetParameterFloat("mode.binary.foreground", 1)
    app.SetParameterString("out", file_out + "?&writerpctags=true&gdal:co:COMPRESS=DEFLATE")
    app.SetParameterOutputImagePixelType("out", output_dtype)
    app.ExecuteAndWriteOutput()

    print("Rasterize in", time.time() - start_time, "seconds.")


def sensor_projection(input_data, sensor_image, dtm_file, geoid_file, projected_data, step=30):
    """ 
    Reproject georeferenced data into sensor geometry

    :param str input_data: path to global data (ie : Pekel, WSF, etc.) to crop and reproject
    :param str sensor_image: path to input image in its raw geometry (sensor)
    :param str dtm_file: path to the DTM
    :param str projected_data: path to the output projected data
    :param int step: TODO document (default, 30)
    """
    from shareloc.geomodels import GeoModel
    from shareloc.dtm_reader import dtm_reader
    from shareloc.geofunctions.dtm_intersection import DTMIntersection
    from shareloc.geofunctions.localization import Localization
    from shareloc.proj_utils import transform_physical_point_to_index
    from shareloc.image import Image

    import bindings_cpp
    import scipy
    
    # Import image geometrical model
    geom_model = GeoModel(sensor_image)
    geom_model_optim = GeoModel(sensor_image, "RPCoptim")

    # Read image and retrieve its bbox coordinates
    data_img = rio.open(sensor_image)
    nb_row, nb_col = data_img.profile['height'], data_img.profile['width']
    transf = data_img.profile['transform']
    start_col, start_row = transf[2], transf[5]
    # DBG : TODO : check if it's normal to take abs
    pix_row, pix_col = np.abs(transf[4]), np.abs(transf[0])
    bbox = np.array([[0, 0], [nb_col + step, 0], [nb_col + step, nb_row + step], [0, nb_row + step]])

    # Resampled image
    x = np.arange(0, nb_col + step, step)
    y = np.arange(0, nb_row + step, step)
    col, row = np.meshgrid(x, y)
    grid_nb_cols, grid_nb_rows = col.shape

    print(f"DBG> {col.shape=} {x=} {y=}")
    
    epi_lp = np.vstack((col.flatten(), row.flatten()))
    epi_pos_left = epi_lp.transpose()
    # Full image
    print(f"DBG> {pix_col=} {pix_row=}")
    all_x = np.arange(0, nb_col, pix_col)
    all_y = np.arange(0, nb_row, pix_row)
    all_col, all_row = np.meshgrid(all_x, all_y)
    all_coords = np.vstack((all_col.flatten(), all_row.flatten())).transpose()
    print(f"DBG> {nb_col=} {pix_col=} {all_x=}")
    print(f"DBG> {all_col=}")
    print(f"DBG> {all_coords.shape=}")
    
    # Load Shareloc direct loc function
    image = Image(sensor_image)
    print(f"DBG> {sensor_image=} {input_data=} {dtm_file=} {geoid_file=}")
    dtm_image = dtm_reader(
        dtm_file,
        geoid_file,
        roi=None,
        roi_is_in_physical_space=True,
        fill_nodata=None,
        fill_value=0.0
    )
    dtm_optim = bindings_cpp.DTMIntersection(
        dtm_image.epsg,
        dtm_image.alt_data,
        dtm_image.nb_rows,
        dtm_image.nb_columns,
        dtm_image.transform
    )
    loc_optim = Localization(geom_model_optim, elevation=dtm_optim, image=image, epsg=4326)
    print(f"DBG> {loc_optim=}")
    # Get bbox pixel coordinates in lat/lon
    coords_bbox_min = None
    coords_bbox_max = None
    alt_min = dtm_optim.get_alt_min()
    alt_max = dtm_optim.get_alt_max()
    print(f"dtm min {alt_min} dtm max {alt_max}")

    for coord in bbox:
        # row /col
        latlon_alt_min = loc_optim.direct(coord[1], coord[0], h=alt_min, using_geotransform=True)
        latlon_alt_max = loc_optim.direct(coord[1], coord[0], h=alt_max, using_geotransform=True)
        coords_bbox_min = latlon_alt_min if coords_bbox_min is None else np.append(coords_bbox_min, latlon_alt_min, axis=0)
        coords_bbox_max = latlon_alt_max if coords_bbox_max is None else np.append(coords_bbox_max, latlon_alt_max, axis=0)
    min_lon =  np.minimum(np.min(coords_bbox_min[:,0]), np.min(coords_bbox_max[:,0]))
    max_lon =  np.maximum(np.max(coords_bbox_min[:,0]), np.max(coords_bbox_max[:,0]))
    min_lat =  np.minimum(np.min(coords_bbox_min[:,1]), np.min(coords_bbox_max[:,1]))
    max_lat =  np.maximum(np.max(coords_bbox_min[:,1]), np.max(coords_bbox_max[:,1]))

    print(f"DBG> {coords_bbox_max=}")
    
    # Direct localization of resampled image to get associated terrain coordinates
    coords_4326 = None
    for pix_x, pix_y in epi_pos_left:
        if coords_4326 is None:
            coords_4326 = np.array([loc_optim.direct(pix_x, pix_y, using_geotransform=True)[0][:2]])
        else:
            coords_4326 = np.append(coords_4326, np.array([loc_optim.direct(pix_x, pix_y, using_geotransform=True)[0][:2]]), axis=0)

    coords_lon = coords_4326[:,0].reshape((grid_nb_cols,grid_nb_rows))
    coords_lat = coords_4326[:,1].reshape((grid_nb_cols,grid_nb_rows))

    print(f"DBG> {coords_lon=} {coords_lat=}")
    
    # interpolate positions on image coordinates
    grid_positions = (y, x)

    # construct all pixels positions
    interp_lon= scipy.interpolate.interpn(
        grid_positions, coords_lon, all_coords, method="linear", bounds_error=False, fill_value=None
    )
    interp_lat = scipy.interpolate.interpn(
        grid_positions, coords_lat, all_coords, method="linear", bounds_error=False, fill_value=None
    )
    interp_pos = np.stack((interp_lat, interp_lon)).transpose()

    # load subset of external data
    roi = [min_lat, min_lon, max_lat, max_lon]
    image_roi = Image(input_data, read_data=True, roi=roi, roi_is_in_physical_space=True)

    #transform positions in pekel image positions
    indexes = transform_physical_point_to_index(image_roi.trans_inv,interp_pos[:,0], interp_pos[:,1])

    # Nearest Neighbor
    # TODO use cars-resample to do linear interpolation ?
    values = np.round(indexes).astype(int)
    # get data at indexes
    image_data = image_roi.data[values[0,:], values[1,:]]

    # DBG
    print(f"DBG> {image_roi.data=}")
    print(f"DBG> {indexes=}")
    print(f"DBG> {values[0,:]=}  {values[1,:]=}")
    print(f"DBG> {image_roi.data.shape=} {image_data.shape=} {nb_row=} {nb_col=}")
    reshaped_image = np.reshape(image_data, (nb_row, nb_col))

    ext_data = rio.open(input_data)
    profile = data_img.profile
    # force GTiff as output
    profile.update({'count': 1, 'dtype': ext_data.profile['dtype'], 'driver': "GTiff"})
    
    dst2 = rio.open(projected_data, 'w', **profile)
    dst2.write(reshaped_image, indexes=1)
    dst2 = None


