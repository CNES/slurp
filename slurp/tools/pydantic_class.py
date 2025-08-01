from pydantic import BaseModel, Field
from typing import get_args, get_origin, Union, List, Optional
import json


# Création d'une fonction qui extrait les informations nécessaires des classes Pydantic
import pandas as pd
from pydantic import BaseModel

def extract_field_info(model_class, prefix=""):
    table_data = []

    for field_name, model_field in model_class.model_fields.items():
        field_type = model_field.annotation
        description = model_field.description or ""
        full_field_name = f"{prefix}.{field_name}" if prefix else field_name

        origin = get_origin(field_type)
        args = get_args(field_type)

        # Si c'est une sous-classe Pydantic directe
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            table_data.append([full_field_name, str(field_type.__name__), description])
            table_data.extend(extract_field_info(field_type, prefix=full_field_name))

        # Si c'est une List[...] ou Optional[...] ou Union[...] contenant une sous-classe Pydantic
        elif origin in [list, List, Union, Optional] and any(
            isinstance(arg, type) and issubclass(arg, BaseModel) for arg in args
        ):
            table_data.append([full_field_name, str(field_type), description])
            for arg in args:
                if isinstance(arg, type) and issubclass(arg, BaseModel):
                    table_data.extend(extract_field_info(arg, prefix=full_field_name))

        else:
            table_data.append([full_field_name, str(field_type), description])

    return table_data

def generate_markdown_table(config_class, file_path):
    table_data = extract_field_info(config_class)
    df = pd.DataFrame(table_data, columns=["Nom du Champ", "Type Attendu", "Description"])

    markdown_table = df.to_markdown(index=False)  # Convert the dataframe to markdown format
    with open(file_path, "w") as f:
        f.write(markdown_table)


class Resources(BaseModel):
    n_workers: int = Field(..., description="Number of CPU")
    tile_max_size: int = Field(..., description="Maximum size of tiles to process")
    multiproc_context: str = Field(default="spawn", description="Multiprocessing strategy: 'fork' or 'spawn' for EOScale")
    n_jobs: int = Field(..., description="Nb of parallel jobs for Random Forest (1 is recommanded : use n_workers to optimize parallel computing)")
    save_mode: str = Field(..., description="Save all files (debug) or only output mask (none)")

class Prepare(BaseModel):
    red: int = Field(..., description="Red band index")
    green: int = Field(..., description="Green band index")
    nir: int = Field(..., description="NIR band index")
    cloud_mask: Optional[str] = Field(None, description="Path to the input cloud mask")
    pekel_method: str = Field(..., description="Method for Pekel recovery : 'all' for global file and 'month' for monthly recovery")
    pekel: Optional[str] = Field(..., description="Path of the global Pekel file")
    pekel_monthly_occurrence: Optional[str] = Field(default=None, description="Path of the root of monthly occurrence Pekel files")
    pekel_obs: Optional[str] = Field(None, description="Month of the desired Pekel file (pekel_method = month)")
    hand: Optional[str] = Field(..., description="Path of the global HAND file")
    wsf: Optional[str] = Field(..., description="Path of the global WSF file")
    texture_rad: int = Field(..., description="Radius for texture (std convolution) computation")
    dtm: Optional[str] = Field(None, description="Digital Terrain Model, used only in sensor mode")
    geoid_file: str = Field(..., description="Geoid file, used only in sensor mode")
    analyse_glcm: bool = Field(..., description="Use a global land cover map to calculate the better number of vegetation cluster to use for mask computation")
    land_cover_map: str = Field(..., description="Input land cover map, only used if 'analyse_glcm' is True")
    cropped_land_cover_map: bool = Field(..., description="If the land_cover_map image is cropped to the input VHR file or not")
    effective_used_config: str = Field(..., description="Path to the effective configuration used")

class PostProcess(BaseModel):
    binary_opening: int = Field(..., description="Size of disk structuring element")
    binary_closing: int = Field(..., description="Size of disk structuring element")
    binary_dilation: int = Field(..., description="Size of disk structuring element")
    remove_small_objects: int = Field(..., description="The maximum area, in pixels, of a contiguous object that will be removed")
    remove_small_holes: int = Field(..., description="The maximum area, in pixels, of a contiguous hole that will be filled")
    area_closing: Optional[str] = Field(None, description="Area closing removes all dark structures")

class Shadows(BaseModel):
    th_rgb: float = Field(..., description="Relative shadow threshold for RGB bands")
    th_nir: float = Field(..., description="Relative shadow threshold for NIR band")
    percentile: int = Field(..., description="Percentile value to cut histogram and estimate shadow threshold")
    absolute_threshold: bool = Field(..., description="Compute shadow mask with a unique absolute threshold")

class Urban(BaseModel):
    files_layers: Optional[List[str]] = Field(default=None, description="Add layers as additional features used by learning algorithm")
    vegmask_min_value: Optional[int] = Field(default=None, description="Vegetation min value for vegetated areas : all pixels with lower value will be predicted")
    veg_binary_dilation: int = Field(..., description="Size of disk structuring element (dilate non vegetated areas)")
    value_classif: int = Field(..., description="Input ground truth class to consider in the input ground truth")
    gt_binary_erosion: int = Field(..., description="Size of disk structuring element (erode GT before picking-up samples)")
    nb_samples_other: int = Field(..., description="Number of samples in other for learning")
    nb_samples_urban: int = Field(..., description="Number of samples in buildings for learning")
    max_depth: int = Field(..., description="Maximum depth of the decision tree")
    nb_estimators: int = Field(..., description="Number of trees in Random Forest")

class Vegetation(BaseModel):
    texture_mode: str = Field(..., description="Labelize vegetation with (yes) or without (no) distinction low/high, or get all vegetation clusters without distinction low/high (debug)")
    filter_texture: int = Field(..., description="Percentile for texture (between 1 and 99)")
    slic_seg_size: int = Field(..., description="Approximative segment size")
    slic_compactness: float = Field(..., description="Balance between color and space proximity (see skimage.slic documentation)")
    nb_clusters_veg: int = Field(..., description="Nb of clusters considered as vegetation (1-NB_CLUSTERS)")
    min_ndvi_veg: Optional[float] = Field(None, description="Minimal mean NDVI value to consider a cluster as vegetation (overload nb clusters choice)")
    max_ndvi_noveg: Optional[float] = Field(None, description="Maximal mean NDVI value to consider a cluster as non-vegetation (overload nb clusters choice)")
    non_veg_clusters: Optional[List[int]] = Field(None, description="Labelize each 'non vegetation cluster' as 0, 1, 2 (..) instead of single label (0)")
    nb_clusters_low_veg: int = Field(..., description="Nb of clusters considered as low vegetation (1-NB_CLUSTERS)")
    max_texture_th: Optional[float] = Field(None, description="Maximal texture value to consider a cluster as low vegetation (overload nb clusters choice)")
    debug: bool = Field(..., description="Flag for debugging mode")

class Water(BaseModel):
    files_layers: List[str] = Field(..., description="Add layers as additional features used by learning algorithm")
    thresh_pekel: int = Field(..., description="Threshold for Pekel water occurrence detection")
    thresh_hand: int = Field(..., description="Hand Threshold int >= 0")
    hand_strict: bool = Field(..., description="Use not(pekelxx) for other (no water) samples")
    strict_thresh: int = Field(..., description="Pekel Threshold float if hand_strict")
    simple_ndwi_threshold: bool = Field(..., description="Compute water mask as a simple NDWI threshold - useful in arid places where no water is known by Peckel")
    ndwi_threshold: float = Field(..., description="Threshold used when Pekel is empty in the area")
    samples_method: str = Field(..., description="Select method for choosing learning samples ('grid' or 'random' or 'smart')")
    nb_samples_water: int = Field(..., description="Number of samples in water for learning")
    nb_samples_other: int = Field(..., description="Number of samples in other for learning")
    nb_samples_auto: bool = Field(..., description="Auto select number of samples for water and other")
    auto_pct: float = Field(..., description="Percentage of samples points, to use with -nb_samples_auto")
    smart_area_pct: int = Field(..., description="For smart method, importance of area for selecting number of samples in each water surface")
    smart_minimum: int = Field(..., description="For smart method, minimum number of samples in each water surface.")
    grid_spacing: int = Field(..., description="For grid method, select samples on a regular grid (40 pixels seems to be a good value)")
    max_depth: int = Field(..., description="Max depth of trees")
    nb_estimators: int = Field(..., description="Number of estimators for the water classifier")
    no_pekel_filter: bool = Field(..., description="Deactivate postprocess with pekel which only keeps surfaces already known by pekel")
    hand_filter: bool = Field(..., description="Postprocess with Hand (set to 0 when hand > thresh), incompatible with hand_strict")
    value_classif: int = Field(..., description="Output classification value (default is 1)")

class StackMain(BaseModel):
    building_threshold: int = Field(..., description="Threshold for building detection in the stack")
    building_erosion: int = Field(..., description="Supposed buildings will be eroded by this size in the marker step")
    bonus_gt: int = Field(..., description="Bonus for pixels covered by GT, in the watershed regularization step (ex : +30 to improve discrimination between building and background)")
    malus_shadow: int = Field(..., description="Value of the malus for pixels in shadow, in the watershed regularization step")
    value_classif_low_veg: int = Field(..., description="Output classification value for low vegetation")
    value_classif_high_veg: int = Field(..., description="Output classification value for high vegetation")
    value_classif_water: int = Field(..., description="Output classification value for water")
    value_classif_buildings: int = Field(..., description="Output classification value for buildings")
    value_classif_bare_ground: int = Field(..., description="Output classification value for bare ground")
    value_classif_false_positive_buildings: int = Field(..., description="Output classification value for buildings false positive")
    value_classif_background: int = Field(..., description="Output classification value for background")

class StackUser(BaseModel):
    vegmask_max_value: int = Field(..., description="Maximum allowed value in the vegetation mask")
    waterpred: str = Field(..., description="Predicted water mask file")
    urban_proba: str = Field(..., description="Probability map for urban areas")
    building_threshold: int = Field(..., description="Threshold for building detection in the stack")
    building_erosion: int = Field(..., description="Supposed buildings will be eroded by this size in the marker step")
    binary_closing: int = Field(..., description="Size of disk structuring element")
    binary_opening: int = Field(..., description="Size of disk structuring element")
    bonus_gt: int = Field(..., description="Bonus for pixels covered by GT, in the watershed regularization step (ex : +30 to improve discrimination between building and background)")
    malus_shadow: int = Field(..., description="Value of the malus for pixels in shadow, in the watershed regularization step")
    remove_small_objects: int = Field(..., description="The maximum area, in pixels, of a contiguous object that will be removed")
    remove_small_holes: int = Field(..., description="The maximum area, in pixels, of a contiguous hole that will be filled")

class Masks(BaseModel):
    watermask: Optional[str] = Field(..., description="For watermask computation: Output classification filename. For stackmask: Water mask to stack. Otherwise, if given, output mask will exclude water areas")
    urbanmask: Optional[str] = Field(..., description="For urbanmask computation: Output classification filename. For stackmask: Vegetation mask to stack.")
    vegetationmask: Optional[str] = Field(..., description="For vegetationmask computation : Output classification filename. For stackmask: Vegetation mask to stack. Otherwise, if given, output mask will exclude vegetation areas")
    shadowmask: Optional[str] = Field(..., description="For shadowmask computation : Output classification filename. For stackmask: Shadow mask to stack. Otherwise, if given, big shadow areas will be marked as background")
    stackmask: Optional[str] = Field(..., description="Output classification filename")

class AuxLayersMain(BaseModel):
    valid_stack: Optional[str] = Field(None, description="Path to store the valid stack file")
    file_ndvi: Optional[str] = Field(..., description="Path to the NDVI layer")
    file_ndwi: Optional[str] = Field(..., description="Path to the NDWI layer")
    extracted_pekel: Optional[str] = Field(..., description="Path to the extracted Pekel data")
    extracted_hand: Optional[str] = Field(..., description="Path to the extracted HAND data")
    extracted_wsf: Optional[str] = Field(..., description="Extracted World Settlement Footprint raster filename")
    file_texture: Optional[str] = Field(..., description="Path to store the texture file")
    mnh: Optional[str] = Field(None, description="Path to the Mean Height of Nearest Neighbors (MN) layer")

class AuxLayersUser(BaseModel):
    extracted_pekel: str = Field(..., description="TIFF file with extracted PEKEL values")
    extracted_hand: str = Field(..., description="TIFF file with extracted HAND values")
    extracted_wsf: Optional[str] = Field(..., description="Extracted World Settlement Footprint raster filename")
    file_ndvi: Optional[str] = Field(None, description="NDVI file")
    file_ndwi: Optional[str] = Field(None, description="NDWI file")
    valid_stack: Optional[str] = Field(None, description="Path to store the valid stack file")
    file_cloud_gml: Optional[str] = Field(None, description="GML file containing cloud masks (optional)")
    file_texture: Optional[str] = Field(..., description="Path to store the texture file")
    mnh: Optional[str] = Field(None, description="Digital surface model (DSM) file (optional)")

class InputMain(BaseModel):
    file_vhr: str = Field(..., description="Input 4 bands VHR (Very High Resolution) image")
    sensor_mode: bool = Field(..., description="True if input image is in its raw (sensor) geometry, False if input image is georeferenced (orthorectification)")

class InputUser(BaseModel):
    file_vhr: str = Field(..., description="Input 4 bands VHR (Very High Resolution) image")
    pekel: str = Field(..., description="Path of the global Pekel file")
    hand: str = Field(..., description="Path of the global HAND file")
    wsf: str = Field(..., description="Path to store the extracted WSF file")


class MainConfig(BaseModel):
    input: InputMain = Field(..., description="Input data for the configuration")
    aux_layers: AuxLayersMain = Field(..., description="Auxiliary layers used in processing")
    masks: Masks = Field(..., description="Masks for different types of land cover and features")
    resources: Resources = Field(..., description="Resources configuration for parallel processing")
    prepare: Prepare = Field(..., description="Preparation settings for preprocessing")
    post_process: PostProcess = Field(..., description="Settings for post-processing")
    shadows: Shadows = Field(..., description="Shadow detection settings")
    urban: Urban = Field(..., description="Urban classification settings")
    vegetation: Vegetation = Field(..., description="Vegetation classification settings")
    water: Water = Field(..., description="Water detection settings")
    stack: StackMain = Field(..., description="Stack processing settings")

class UserConfig(BaseModel):
    input: InputUser = Field(..., description="Input data for the configuration")
    aux_layers: AuxLayersUser = Field(..., description="Auxiliary layers used in processing")
    masks: Masks = Field(..., description="Masks for different types of land cover and features")
    stack: StackUser = Field(..., description="Stack processing settings")


# Main function for loading the JSON file with Pydantic
def load_config(file_path: str, config_class: BaseModel) -> BaseModel:
    with open(file_path, 'r') as f:
        data = json.load(f)
        return config_class.parse_obj(data)


# Generate markdown table from pydantic class
# generate_markdown_table(MainConfig, "docs/source/main_config_descr.md")
# generate_markdown_table(UserConfig, "docs/source/user_config_descr.md")
