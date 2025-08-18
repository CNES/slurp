import os

import pytest
import rasterio
import numpy as np

from tests.utils import get_files_to_process, get_output_path

def compare_tif(file1, file2, atol=0.0):
    with rasterio.open(file1) as src1, rasterio.open(file2) as src2:
        # Compare metadata
        if src1.meta != src2.meta:
            raise ValueError(f"TIFF metadata differs.")

        # Compare pixel data
        arr1 = src1.read()
        arr2 = src2.read()
        if not np.allclose(arr1, arr2, atol=atol):
            raise ValueError(f"TIFF pixel values differ.")

    print("TIFF files are identical.")
    return True


# Test for slurp_prepare
def test_noreg_slurp_prepare():
    # Input images
    file = get_files_to_process("vegetation")[0]

    valid_stack = get_output_path(file, "valid_stack", remove=True)
    ndvi = get_output_path(file, "ndvi", remove=True)
    ndwi = get_output_path(file, "ndwi", remove=True)
    texture = get_output_path(file, "texture", remove=True)

    print(f"slurp_prepare {pytest.main_config} -file_vhr {file} -n_workers 1 "
    f"-valid {valid_stack} -file_ndvi {ndvi} -file_ndwi {ndwi} -file_texture {texture} "
    f"-mode vegetation --analyse_glcm")

    os.system(
        f"slurp_prepare {pytest.main_config} -file_vhr {file} -n_workers 1 "
        f"-valid {valid_stack} -file_ndvi {ndvi} -file_ndwi {ndwi} -file_texture {texture} "
        f"-mode vegetation --analyse_glcm"
    )

    assert os.path.exists(
        valid_stack
    ), f"The file {valid_stack} has not been created. Error during valid stack computation ?"
    assert os.path.exists(
        ndvi
    ), f"The file {ndvi} has not been created. Error during NDVI computation ?"
    assert os.path.exists(
        ndwi
    ), f"The file {ndwi} has not been created. Error during NDWI computation ?"
    assert os.path.exists(
        texture
    ), f"The file {texture} has not been created. Error during Texture computation ?"

    ######## COMPARE OUTPUT FILES WITH REFERENCE
    compare_tif("out/ref_prepare/ndvi_xt_zone_NO_DATA.tif", "out/ndvi_xt_zone_NO_DATA.tif")
    compare_tif("out/ref_prepare/ndwi_xt_zone_NO_DATA.tif", "out/ndwi_xt_zone_NO_DATA.tif")
    compare_tif("out/ref_prepare/texture_xt_zone_NO_DATA.tif", "out/texture_xt_zone_NO_DATA.tif")
    compare_tif("out/ref_prepare/valid_stack_xt_zone_NO_DATA.tif", "out/valid_stack_xt_zone_NO_DATA.tif")
    return valid_stack, ndvi, ndwi, texture

