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

def generate_markdown_table(config_class):
    table_data = extract_field_info(config_class)
    df = pd.DataFrame(table_data, columns=["Nom du Champ", "Type Attendu", "Description"])

    df.to_csv("pydantic_class_filled.csv")
    return df


class Resources(BaseModel):
    n_workers: int = Field(..., description="Number of worker threads for parallel processing")
    tile_max_size: int = Field(..., description="Maximum size of tiles to process")
    multiproc_context: str = Field(..., description="Multiprocessing context type (e.g., 'fork')")
    n_jobs: int = Field(..., description="Number of jobs to run in parallel")
    save_mode: str = Field(..., description="Save mode (e.g., 'none', 'all', 'partial')")

class Prepare(BaseModel):
    red: int = Field(..., description="Index for the red band in the input image")
    green: int = Field(..., description="Index for the green band in the input image")
    nir: int = Field(..., description="Index for the near-infrared (NIR) band in the input image")
    cloud_mask: Optional[str] = Field(None, description="Path to the cloud mask file")
    pekel_method: str = Field(..., description="Method for Pekel water occurrence detection")
    pekel: Optional[str] = Field(..., description="Path to the Pekel occurrence data")
    pekel_monthly_occurrence: Optional[str] = Field(default=None, description="Path to the monthly occurrence data for Pekel")
    pekel_obs: Optional[str] = Field(None, description="Path to Pekel observation data")
    hand: Optional[str] = Field(..., description="Path to the Height Above the Nearest Drainage (HAND) model")
    wsf: Optional[str] = Field(..., description="Path to the World Settlement Footprint (WSF) data")
    texture_rad: int = Field(..., description="Texture radius for the texture analysis")
    dtm: Optional[str] = Field(None, description="Path to the Digital Terrain Model (DTM) file")
    geoid_file: str = Field(..., description="Path to the geoid file")
    analyse_glcm: bool = Field(..., description="Flag to enable or disable GLCM texture analysis")
    land_cover_map: str = Field(..., description="Path to the land cover classification map")
    cropped_land_cover_map: bool = Field(..., description="Flag to crop the land cover map")
    effective_used_config: str = Field(..., description="Path to the effective configuration used")

class PostProcess(BaseModel):
    binary_opening: int = Field(..., description="Size of the binary opening operation")
    binary_closing: int = Field(..., description="Size of the binary closing operation")
    binary_dilation: int = Field(..., description="Size of the binary dilation operation")
    remove_small_objects: int = Field(..., description="Minimum size of objects to remove in post-processing")
    remove_small_holes: int = Field(..., description="Maximum size of holes to remove in post-processing")
    area_closing: Optional[str] = Field(None, description="Area closing threshold for small regions")

class Shadows(BaseModel):
    th_rgb: float = Field(..., description="Threshold for shadow detection in RGB bands")
    th_nir: float = Field(..., description="Threshold for shadow detection in the NIR band")
    percentile: int = Field(..., description="Percentile for shadow detection threshold")
    absolute_threshold: bool = Field(..., description="Flag to use absolute threshold for shadow detection")

class Urban(BaseModel):
    files_layers: Optional[List[str]] = Field(default=None, description="List of file paths to urban layers")
    vegmask_min_value: Optional[int] = Field(default=None, description="Minimum value for vegetation mask in urban analysis")
    veg_binary_dilation: int = Field(..., description="Binary dilation size for the vegetation mask")
    value_classif: int = Field(..., description="Value for classification of urban areas")
    gt_binary_erosion: int = Field(..., description="Binary erosion size for ground truth")
    nb_samples_other: int = Field(..., description="Number of samples for other areas in urban classification")
    nb_samples_urban: int = Field(..., description="Number of samples for urban areas in urban classification")
    max_depth: int = Field(..., description="Maximum depth of the decision tree for urban classification")
    nb_estimators: int = Field(..., description="Number of estimators for the urban classifier")

class Vegetation(BaseModel):
    texture_mode: str = Field(..., description="Texture analysis mode ('yes' or 'no')")
    filter_texture: int = Field(..., description="Filter size for texture analysis")
    slic_seg_size: int = Field(..., description="Size of the SLIC superpixels for segmentation")
    slic_compactness: float = Field(..., description="Compactness parameter for SLIC segmentation")
    nb_clusters_veg: int = Field(..., description="Number of vegetation clusters for classification")
    min_ndvi_veg: Optional[float] = Field(None, description="Minimum NDVI value for vegetation classification")
    max_ndvi_noveg: Optional[float] = Field(None, description="Maximum NDVI value for non-vegetation classification")
    non_veg_clusters: Optional[List[int]] = Field(None, description="Cluster indices representing non-vegetation areas")
    nb_clusters_low_veg: int = Field(..., description="Number of low vegetation clusters for classification")
    max_texture_th: Optional[float] = Field(None, description="Maximum threshold for texture classification")
    debug: bool = Field(..., description="Flag for debugging mode")

class Water(BaseModel):
    files_layers: List[str] = Field(..., description="List of file paths for water-related layers")
    thresh_pekel: int = Field(..., description="Threshold for Pekel water occurrence detection")
    thresh_hand: int = Field(..., description="Threshold for HAND-based water detection")
    hand_strict: bool = Field(..., description="Strictness flag for HAND-based water detection")
    strict_thresh: int = Field(..., description="Strictness threshold for water detection")
    simple_ndwi_threshold: bool = Field(..., description="Flag to apply a simple NDWI threshold for water detection")
    ndwi_threshold: float = Field(..., description="NDWI threshold value for water classification")
    samples_method: str = Field(..., description="Sampling method for water detection ('grid' or 'random')")
    nb_samples_water: int = Field(..., description="Number of samples for water detection")
    nb_samples_other: int = Field(..., description="Number of samples for other types of land cover")
    nb_samples_auto: bool = Field(..., description="Flag to enable automatic sample selection")
    auto_pct: float = Field(..., description="Percentage of the total area for automatic sample selection")
    smart_area_pct: int = Field(..., description="Percentage of smart area for water detection")
    smart_minimum: int = Field(..., description="Minimum area size for smart water detection")
    grid_spacing: int = Field(..., description="Spacing between grid points for water detection")
    max_depth: int = Field(..., description="Maximum depth for decision tree models in water classification")
    nb_estimators: int = Field(..., description="Number of estimators for the water classifier")
    no_pekel_filter: bool = Field(..., description="Flag to disable Pekel filtering in water detection")
    hand_filter: bool = Field(..., description="Flag to enable HAND filtering for water detection")
    value_classif: int = Field(..., description="Classification value for water areas")

class StackMain(BaseModel):
    building_threshold: int = Field(..., description="Threshold for building detection in the stack")
    building_erosion: int = Field(..., description="Erosion size for building detection")
    bonus_gt: int = Field(..., description="Bonus to add for ground truth buildings")
    malus_shadow: int = Field(..., description="Malus for shadow detection")
    value_classif_low_veg: int = Field(..., description="Classification value for low vegetation areas")
    value_classif_high_veg: int = Field(..., description="Classification value for high vegetation areas")
    value_classif_water: int = Field(..., description="Classification value for water areas")
    value_classif_buildings: int = Field(..., description="Classification value for buildings")
    value_classif_bare_ground: int = Field(..., description="Classification value for bare ground")
    value_classif_false_positive_buildings: int = Field(..., description="Classification value for false positive buildings")
    value_classif_background: int = Field(..., description="Classification value for background areas")

class StackUser(BaseModel):
    vegmask_max_value: int = Field(..., description="Maximum allowed value in the vegetation mask")
    waterpred: str = Field(..., description="Predicted water mask file")
    urban_proba: str = Field(..., description="Probability map for urban areas")
    building_threshold: int = Field(..., description="Threshold value for building detection")
    binary_closing: int = Field(..., description="Binary morphological closing parameter")
    binary_opening: int = Field(..., description="Binary morphological opening parameter")
    building_erosion: int = Field(..., description="Erosion value applied to buildings")
    bonus_gt: int = Field(..., description="Bonus value applied to ground truth pixels")
    malus_shadow: int = Field(..., description="Penalty applied to shadow areas")
    remove_small_objects: int = Field(..., description="Flag to remove small objects (1 = yes, 0 = no)")
    remove_small_holes: int = Field(..., description="Flag to fill small holes (1 = yes, 0 = no)")

class Masks(BaseModel):
    watermask: Optional[str] = Field(..., description="Path to the water mask file")
    urbanmask: Optional[str] = Field(..., description="Path to the urban mask file")
    vegetationmask: Optional[str] = Field(..., description="Path to the vegetation mask file")
    shadowmask: Optional[str] = Field(..., description="Path to the shadow mask file")
    stackmask: Optional[str] = Field(..., description="Path to the stack mask file")

class AuxLayersMain(BaseModel):
    valid_stack: Optional[str] = Field(..., description="Path to the valid stack layer")
    file_ndvi: Optional[str] = Field(..., description="Path to the NDVI layer")
    file_ndwi: Optional[str] = Field(..., description="Path to the NDWI layer")
    extracted_pekel: Optional[str] = Field(..., description="Path to the extracted Pekel data")
    extracted_hand: Optional[str] = Field(..., description="Path to the extracted HAND data")
    extracted_wsf: Optional[str] = Field(..., description="Path to the extracted WSF data")
    file_texture: Optional[str] = Field(..., description="Path to the texture layer")
    mnh: Optional[str] = Field(None, description="Path to the Mean Height of Nearest Neighbors (MN) layer")

class AuxLayersUser(BaseModel):
    extracted_pekel: str = Field(..., description="TIFF file with extracted PEKEL values")
    extracted_hand: str = Field(..., description="TIFF file with extracted HAND values")
    extracted_wsf: str = Field(..., description="TIFF file with extracted WSF values")
    file_ndvi: Optional[str] = Field(None, description="NDVI file (may be null if unavailable)")
    file_ndwi: Optional[str] = Field(None, description="NDWI file (may be null if unavailable)")
    valid_stack: Optional[str] = Field(None, description="Validated image stack (may be null)")
    file_cloud_gml: Optional[str] = Field(None, description="GML file containing cloud masks (optional)")
    file_texture: Optional[str] = Field(None, description="Texture image file (optional)")
    mnh: Optional[str] = Field(None, description="Digital surface model (DSM) file (optional)")

class InputMain(BaseModel):
    file_vhr: str = Field(..., description="Path to the input VHR (Very High Resolution) image file")
    sensor_mode: bool = Field(..., description="Flag to enable or disable sensor mode")

class InputUser(BaseModel):
    file_vhr: str = Field(..., description="Path to the VHR satellite image")
    pekel: str = Field(..., description="VRT file containing PEKEL water occurrence data")
    hand: str = Field(..., description="HAND model derived from MERIT DEM")
    wsf: str = Field(..., description="Probability map of built-up areas (WSF 2019)")


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


# Fonction principale pour charger le fichier JSON avec Pydantic
def load_main_config(file_path: str) -> 'MainConfig':
    with open(file_path, 'r') as f:
        # Charger le JSON et le parser directement avec Config.parse_obj ou Config.parse_raw
        data = json.load(f)
        return MainConfig.parse_obj(data)

def load_user_config(file_path: str) -> 'UserConfig':
    with open(file_path, 'r') as f:
        # Charger le JSON et le parser directement avec Config.parse_obj ou Config.parse_raw
        data = json.load(f)
        return UserConfig.parse_obj(data)


# Génération du tableau markdown à partir de la classe Config
markdown_table = generate_markdown_table(MainConfig)

# Affichage du tableau
print(markdown_table)