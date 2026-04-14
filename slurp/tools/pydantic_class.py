import json
from typing import List, Optional

from pydantic import BaseModel, Field


class Resources(BaseModel):
    debug: Optional[bool] = Field(
        default=False, description="Flag for debugging mode (default : False)"
    )
    n_workers: Optional[int] = Field(default=8, description="Number of CPU")
    tile_max_size: Optional[int] = Field(
        default=0,
        description="Maximum size of tiles to process (0 : let EOScale choose max size per CPU",
    )
    multiproc_context: Optional[str] = Field(
        default="spawn",
        description="Multiprocessing strategy: 'fork' or 'spawn' for EOScale",
    )
    n_jobs: Optional[int] = Field(
        default=1,
        description="Nb of parallel jobs for Random Forest (1 is recommanded :"
        "use n_workers to optimize parallel computing)",
    )
    save_mode: Optional[str] = Field(
        default=None,
        description="Save all files (debug) or only output mask (none)",
    )


class Prepare(BaseModel):
    red: Optional[int] = Field(default=1, description="Red band index")
    green: Optional[int] = Field(default=2, description="Green band index")
    nir: Optional[int] = Field(default=4, description="NIR band index")
    cloud_mask: Optional[str] = Field(
        None, description="Path to the input cloud mask"
    )
    pekel_method: Optional[str] = Field(
        default="all",
        description="Method for Pekel recovery : 'all' for global file and 'month' "
        "for monthly recovery",
    )
    pekel: Optional[str] = Field(
        None, description="Path of the global Pekel (Global Surface Water) file"
    )
    pekel_monthly_occurrence: Optional[str] = Field(
        default=None,
        description="Path of the root of monthly occurrence Pekel files",
    )
    pekel_obs: Optional[str] = Field(
        None,
        description="Month of the desired Pekel (Global Surface Water)"
        " file (pekel_method = month)",
    )
    hand: Optional[str] = Field(
        None,
        description="Path of the global Height Above Nearest Drainage (HAND) file",
    )
    wsf: Optional[str] = Field(
        None,
        description="Path of the global World Settlement Footprint (WSF) file",
    )
    wbm: Optional[str] = Field(
        None, description="Path of the global Water Body Mask (WBM) file"
    )
    texture_rad: Optional[int] = Field(
        default=5,
        description="Radius for texture (std convolution) computation",
    )
    dtm: Optional[str] = Field(
        None, description="Digital Terrain Model, used only in sensor mode"
    )
    geoid_file: Optional[str] = Field(
        None, description="Geoid file, used only in sensor mode"
    )
    analyse_glcm: Optional[bool] = Field(
        default=False,
        description="Use a global land cover map to calculate the better "
        "number of vegetation cluster to use for mask computation",
    )
    land_cover_map: Optional[str] = Field(
        None,
        description="Input land cover map, only used if 'analyse_glcm' is True",
    )
    cropped_land_cover_map: Optional[bool] = Field(
        None,
        description="If the land_cover_map image is cropped to the input VHR file or not",
    )
    effective_used_config: Optional[str] = Field(
        "out/effective_used_config.json",
        description="Path to the effective configuration used",
    )


class PostProcess(BaseModel):
    binary_opening: Optional[int] = Field(
        default=2, description="Size of disk structuring element"
    )
    binary_closing: Optional[int] = Field(
        default=2, description="Size of disk structuring element"
    )
    binary_dilation: Optional[int] = Field(
        default=2, description="Size of disk structuring element"
    )
    remove_small_objects: Optional[int] = Field(
        default=100,
        description="The maximum area, in pixels, of a contiguous object that will be removed",
    )
    remove_small_holes: Optional[int] = Field(
        default=100,
        description="The maximum area, in pixels, of a contiguous hole that will be filled",
    )
    area_closing: Optional[str] = Field(
        None, description="Area closing removes all dark structures"
    )


class Shadows(BaseModel):
    th_rgb: Optional[float] = Field(
        default=0.2, description="Relative shadow threshold for RGB bands"
    )
    th_nir: Optional[float] = Field(
        default=0.2, description="Relative shadow threshold for NIR band"
    )
    percentile: Optional[int] = Field(
        default=2,
        description="Percentile value to compute histogram (cut extreme values) and "
        "estimate shadow threshold",
    )
    absolute_threshold: Optional[bool] = Field(
        default=False,
        description="Compute shadow mask with a unique absolute threshold",
    )


class Urban(BaseModel):
    files_layers: Optional[List[str]] = Field(
        default=[],
        description="Add layers as additional features used by learning algorithm",
    )
    vegmask_min_value: Optional[int] = Field(
        default=21,
        description="Vegetation min value for vegetated areas : all pixels with lower "
        "value will be predicted",
    )
    veg_binary_dilation: Optional[int] = Field(
        default=5,
        description="Size of disk structuring element (dilate non vegetated areas)",
    )
    value_classif: Optional[int] = Field(
        default=255,
        description="Input ground truth class to consider in the input ground truth",
    )
    gt_binary_erosion: Optional[int] = Field(
        default=5,
        description="Size of disk structuring element (erode GT before picking-up samples)",
    )
    nb_samples_other: Optional[int] = Field(
        default=5000, description="Number of samples in other for learning"
    )
    nb_samples_urban: Optional[int] = Field(
        default=2000, description="Number of samples in buildings for learning"
    )
    max_depth: Optional[int] = Field(
        default=8, description="Maximum depth of the decision tree"
    )
    nb_estimators: Optional[int] = Field(
        default=100, description="Number of trees in Random Forest"
    )


class Vegetation(BaseModel):
    texture_mode: Optional[str] = Field(
        default="yes",
        description="Labelize vegetation with (yes) or without (no) distinction low/high,"
        " or get all vegetation clusters without distinction low/high (debug)",
    )
    filter_texture: Optional[int] = Field(
        default=90, description="Percentile for texture (between 1 and 99)"
    )
    slic_seg_size: Optional[int] = Field(
        default=100, description="Approximative segment size"
    )
    slic_compactness: Optional[float] = Field(
        default=0.1,
        description="Balance between color and space proximity "
        "(see skimage.slic documentation)",
    )
    nb_clusters_veg: Optional[int] = Field(
        default=3,
        description="Nb of clusters considered as vegetation (1-NB_CLUSTERS)",
    )
    min_ndvi_veg: Optional[float] = Field(
        None,
        description="Minimal mean NDVI value to consider a cluster as vegetation "
        "(overload nb clusters choice)",
    )
    max_ndvi_noveg: Optional[float] = Field(
        None,
        description="Maximal mean NDVI value to consider a cluster as non-vegetation"
        " (overload nb clusters choice)",
    )
    non_veg_clusters: Optional[List[int]] = Field(
        None,
        description="Labelize each 'non vegetation cluster' as 0, 1, 2 (..) instead "
        "of single label (0)",
    )
    nb_clusters_low_veg: Optional[int] = Field(
        default=3,
        description="Nb of clusters considered as low vegetation (1-NB_CLUSTERS)",
    )
    max_texture_th: Optional[float] = Field(
        None,
        description="Maximal texture value to consider a cluster as low vegetation "
        "(overload nb clusters choice)",
    )
    pct_veg: Optional[float] = Field(
        default=0.0,
        description="Pourcentage of vegetation pixels in the global land cover map",
    )
    pct_low_veg: Optional[float] = Field(
        default=0.0,
        description="Pourcentage of low vegetation pixels in the global land cover map",
    )
    pct_high_veg: Optional[float] = Field(
        default=0.0,
        description="Pourcentage of high vegetation pixels in the global land cover map",
    )
    pct_non_veg: Optional[float] = Field(
        default=0.0,
        description="Pourcentage of non vegetation pixels in the global land cover map",
    )


class Water(BaseModel):
    files_layers: Optional[List[str]] = Field(
        default=[],
        description="Add layers as additional features used by learning algorithm",
    )
    thresh_pekel: Optional[int] = Field(
        default=50, description="Threshold for Pekel water occurrence detection"
    )
    thresh_hand: Optional[int] = Field(
        default=25, description="Hand Threshold int >= 0"
    )
    hand_strict: Optional[bool] = Field(
        default=False,
        description="Use not(pekelxx) for other (no water) samples",
    )
    strict_thresh: Optional[int] = Field(
        default=50, description="Pekel Threshold float if hand_strict"
    )
    simple_ndwi_threshold: Optional[bool] = Field(
        default=False,
        description="Compute water mask as a simple NDWI threshold - useful in arid places "
        "where no water is known by Peckel",
    )
    ndwi_threshold: Optional[float] = Field(
        default=100,
        description="Threshold used when Pekel is empty in the area",
    )
    samples_method: Optional[str] = Field(
        default="grid",
        description="Select method for choosing learning samples "
        "('grid' or 'random' or 'smart')",
    )
    nb_samples_water: Optional[int] = Field(
        default=2000, description="Number of samples in water for learning"
    )
    nb_samples_other: Optional[int] = Field(
        default=5000, description="Number of samples in other for learning"
    )
    nb_samples_auto: Optional[bool] = Field(
        default=False,
        description="Auto select number of samples for water and other",
    )
    auto_pct: Optional[float] = Field(
        default=0.0002,
        description="Percentage of samples points, to use with -nb_samples_auto",
    )
    smart_area_pct: Optional[int] = Field(
        default=50,
        description="For smart method, importance of area for selecting number of "
        "samples in each water surface",
    )
    smart_minimum: Optional[int] = Field(
        default=10,
        description="For smart method, minimum number of samples in each water surface.",
    )
    grid_spacing: Optional[int] = Field(
        default=40,
        description="For grid method, select samples on a regular grid (40 pixels seems "
        "to be a good value)",
    )
    max_depth: Optional[int] = Field(
        default=8, description="Max depth of trees"
    )
    nb_estimators: Optional[int] = Field(
        default=100, description="Number of estimators for the water classifier"
    )
    no_pekel_filter: Optional[bool] = Field(
        default=False,
        description="Deactivate postprocess with pekel which only keeps surfaces already "
        "known by pekel",
    )
    hand_filter: Optional[bool] = Field(
        default=True,
        description="Postprocess with Hand (set to 0 when hand > thresh), incompatible "
        "with hand_strict",
    )
    value_classif: Optional[int] = Field(
        default=1, description="Output classification value (default is 1)"
    )


class Stack(BaseModel):
    building_threshold: Optional[int] = Field(
        default=70, description="Threshold for building detection in the stack"
    )
    building_erosion: Optional[int] = Field(
        default=2,
        description="Supposed buildings will be eroded by this radius in the marker step",
    )
    erosion_radius: Optional[int] = Field(
        default=2,
        description="All classes except buildings will be eroded by this radius in "
        "the marker step",
    )
    bonus_gt: Optional[int] = Field(
        default=10,
        description="Bonus for pixels covered by GT, in the watershed regularization step "
        "(ex : +30 to improve discrimination between building and background)",
    )
    malus_shadow: Optional[int] = Field(
        default=10,
        description="Value of the malus for pixels in shadow, in the watershed "
        "regularization step",
    )
    value_classif_low_veg: Optional[int] = Field(
        default=1, description="Output classification value for low vegetation"
    )
    value_classif_high_veg: Optional[int] = Field(
        default=2, description="Output classification value for high vegetation"
    )
    value_classif_water: Optional[int] = Field(
        default=3, description="Output classification value for water"
    )
    value_classif_buildings: Optional[int] = Field(
        default=4, description="Output classification value for buildings"
    )
    value_classif_bare_ground: Optional[int] = Field(
        default=6, description="Output classification value for bare ground"
    )
    value_classif_sea: Optional[int] = Field(
        default=7, description="Output classification value for sea"
    )
    value_classif_lake: Optional[int] = Field(
        default=8, description="Output classification value for sea"
    )
    value_classif_river: Optional[int] = Field(
        default=9, description="Output classification value for sea"
    )
    value_classif_false_positive_buildings: Optional[int] = Field(
        default=10,
        description="Output classification value for buildings false positive",
    )
    value_classif_background: Optional[int] = Field(
        default=11, description="Output classification value for background"
    )
    vegmask_min_value: Optional[int] = Field(
        default=21,
        description="All classes above that value shall be considered vegetation",
    )
    binary_closing: Optional[int] = Field(
        default=2, description="Size of disk structuring element"
    )
    binary_opening: Optional[int] = Field(
        default=2, description="Size of disk structuring element"
    )
    categorized_watermask: Optional[bool] = Field(
        default=False,
        description="If true, stack_mask will infer water body category "
        "(lake, river, sea, unknown) from a general water body mask",
    )
    minimal_size_water_area: Optional[int] = Field(
        default=10000, description="Minimal area (in pixels) of water bodies"
    )
    margin: Optional[int] = Field(
        default=0,
        description="Margin (in pixels) for parallel computing during final stack merging",
    )


class Masks(BaseModel):
    watermask: Optional[str] = Field(
        default="out/watermask.tif",
        description="For watermask computation: Output classification filename. "
        "For stackmask: Water mask to stack. Otherwise, if given, output mask will "
        "exclude water areas",
    )
    urbanmask: Optional[str] = Field(
        default="out/urbanmask.tif",
        description="For urbanmask computation: Output classification filename. "
        "For stackmask: Vegetation mask to stack.",
    )
    vegetationmask: Optional[str] = Field(
        default="out/vegetationmask.tif",
        description="For vegetationmask computation : Output classification filename. "
        "For stackmask: Vegetation mask to stack. Otherwise, if given, output mask"
        " will exclude vegetation areas",
    )
    shadowmask: Optional[str] = Field(
        default="out/shadowmask.tif",
        description="For shadowmask computation : Output classification filename. "
        "For stackmask: Shadow mask to stack. Otherwise, if given, big shadow areas "
        "will be marked as background",
    )
    stackmask: Optional[str] = Field(
        default="out/stackmask.tif",
        description="Output classification filename",
    )


class AuxLayers(BaseModel):
    valid_stack: Optional[str] = Field(
        default="out/valid_stack.tif",
        description="Path to store the valid stack file",
    )
    file_ndvi: Optional[str] = Field(
        default="out/ndvi.tif", description="Path to the NDVI layer"
    )
    file_ndwi: Optional[str] = Field(
        default="out/ndwi.tif", description="Path to the NDWI layer"
    )
    extracted_pekel: Optional[str] = Field(
        default="out/pekel.tif", description="Path to the extracted Pekel data"
    )
    extracted_hand: Optional[str] = Field(
        default="out/hand.tif", description="Path to the extracted HAND data"
    )
    extracted_wsf: Optional[str] = Field(
        default="out/wsf.tif",
        description="Path to the extracted World Settlement Footprint",
    )
    extracted_wbm: Optional[str] = Field(
        default="out/wbm.tif",
        description="Path to the extracted Water Body Mask",
    )
    file_texture: Optional[str] = Field(
        default="out/texture.tif", description="Path to store the texture file"
    )
    mnh: Optional[str] = Field(
        None,
        description="Path to the Mean Height of Nearest Neighbors (MN) layer",
    )
    file_cloud_gml: Optional[str] = Field(
        None, description="GML file containing cloud masks (optional)"
    )


class Input(BaseModel):
    file_vhr: str = Field(
        ..., description="Input 4 bands VHR (Very High Resolution) image"
    )
    sensor_mode: Optional[bool] = Field(
        default=False,
        description="True if input image is in its raw (sensor) geometry, "
        "False if input image is georeferenced (orthorectification)",
    )


class MainConfig(BaseModel):
    input: Input = Field(..., description="Input data for the configuration")
    aux_layers: AuxLayers = Field(
        ..., description="Auxiliary layers used in processing"
    )
    masks: Masks = Field(
        ..., description="Masks for different types of land cover and features"
    )
    resources: Resources = Field(
        ..., description="Resources configuration for parallel processing"
    )
    prepare: Prepare = Field(
        ..., description="Preparation settings for preprocessing"
    )
    post_process: PostProcess = Field(
        ..., description="Settings for post-processing"
    )
    shadows: Shadows = Field(..., description="Shadow detection settings")
    urban: Urban = Field(..., description="Urban classification settings")
    vegetation: Vegetation = Field(
        ..., description="Vegetation classification settings"
    )
    water: Water = Field(..., description="Water detection settings")
    stack: Stack = Field(..., description="Stack processing settings")


# Main function for loading the JSON file with Pydantic
def load_config(file_path: str, config_class: BaseModel) -> BaseModel:
    with open(file_path, "r") as f:
        data = json.load(f)
        return config_class.parse_obj(data)


# Generate markdown table from pydantic class
# generate_markdown_table(MainConfig, "docs/source/main_config_descr.md")
