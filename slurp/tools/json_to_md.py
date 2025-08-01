from pathlib import Path
from typing import Any, List, Dict, Tuple, Type
import pandas as pd
import mkdocs.plugins
from pydantic import BaseModel, Field
from pydantic_class import MainConfig, UserConfig, InputUser, AuxLayersUser, Masks  # Import your Pydantic models


def sort_fields(schema: Dict[str, Any]) -> Tuple[
    List[str], List[str], List[str]]:
    """
    Sorts the fields of a schema into mandatory, recommended, and optional categories.
    """
    mandatory_fields = []
    recommended_fields = []
    optional_fields = []
    for field_name, meta in schema["properties"].items():
        if "default" not in meta:
            mandatory_fields.append(field_name)
        if "recommended" in meta:
            recommended_fields.append(field_name)
        elif "default" in meta:
            optional_fields.append(field_name)
    mandatory_fields = sorted(mandatory_fields)
    recommended_fields = sorted(recommended_fields)
    optional_fields = sorted(optional_fields)
    return mandatory_fields, recommended_fields, optional_fields


def resolve_ref(ref: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolves a JSON reference ('$ref') in the schema.
    """
    ref_path = ref.split("/")[1:]
    resolved = schema
    for part in ref_path:
        resolved = resolved[part]
    return resolved


def parse_field_metadata(field_name: str, field_status: str, field_meta: Dict[str, Any],
                         schema: Dict[str, Any], fields_df: pd.DataFrame) -> None:
    """
    Parses metadata of a specific field from a schema and adds the relevant information to a DataFrame.
    """
    selectable_values = ""
    if "enum" in field_meta:
        selectable_values = ", ".join(field_meta["enum"])
    elif "items" in field_meta and "enum" in field_meta["items"]:
        selectable_values = ", ".join(field_meta["items"]["enum"])
    elif "anyOf" in field_meta:
        for val in field_meta["anyOf"]:
            if "enum" in val:
                selectable_values = ", ".join(val["enum"])
            if "items" in val and "enum" in val["items"]:
                selectable_values = ", ".join(val["items"]["enum"])
            if "$ref" in val:
                fields_df.loc[len(fields_df.index)] = [field_name, field_meta["description"],
                                                       selectable_values, field_status]
                referenced_meta_schema = resolve_ref(val["$ref"], schema)
                mandatory_fields, recommended_fields, optional_fields = sort_fields(
                    referenced_meta_schema)
                for status, fields in [("**mandatory**", mandatory_fields),
                                       ("**recommended/optional**", recommended_fields),
                                       ("**optional**", optional_fields)]:
                    for field in fields:
                        field_meta = referenced_meta_schema["properties"][field]
                        parse_field_metadata(f"{field_name}.{field}",
                                             status,
                                             field_meta,
                                             referenced_meta_schema,
                                             fields_df)
                return

    # Detect if the field is a Pydantic model (e.g., InputUser, AuxLayersUser, Masks)
    if isinstance(field_meta.get("type"), type) and issubclass(field_meta["type"], BaseModel):
        # If it's a Pydantic model, list the fields inside this model
        model_class = field_meta["type"]
        nested_schema = model_class.schema()  # Get the schema for the model

        # Add the model class name to the metadata
        fields_df.loc[len(fields_df.index)] = [field_name,
                                               f"{field_meta['description']} (Pydantic class: {model_class.__name__})",
                                               selectable_values, field_status]

        # Recursively process the fields of the nested Pydantic model
        for nested_field, nested_meta in nested_schema["properties"].items():
            parse_field_metadata(f"{field_name}.{nested_field}",
                                 "nested",
                                 nested_meta,
                                 nested_schema,
                                 fields_df)

    else:
        fields_df.loc[len(fields_df.index)] = [field_name, field_meta["description"],
                                               selectable_values, field_status]


def fields_descriptions(fields_status: List[Tuple[str, List[str]]],
                        schema: Dict[str, Any]) -> pd.DataFrame:
    """
    Creates a DataFrame containing field metadata descriptions and status.
    """
    fields_df = pd.DataFrame(
        columns=["Metadata", "Description", "selectable values", "Status"])
    for status, fields in fields_status:
        for field in fields:
            field_meta = schema["properties"][field]
            parse_field_metadata(field, status, field_meta, schema, fields_df)
    return fields_df


def schema_to_md(schema: Dict[str, Any], output_md_file: Path) -> None:
    """
    Converts a schema to a Markdown table and saves it to a file.
    """
    fields_df_desc = fields_to_df(schema)
    fields_df_desc.to_markdown(buf=output_md_file, index=False)


def fields_to_df(schema):
    """
    Converts a schema to a DataFrame
    """
    mandatory_fields, recommended_fields, optional_fields = sort_fields(schema)
    fields_df_desc = fields_descriptions(
        [("**mandatory**", mandatory_fields), ("**recommended/optional**", recommended_fields),
         ("**optional**", optional_fields)],
        schema)
    return fields_df_desc


@mkdocs.plugins.event_priority(-50)
def on_pre_build(*args, **kwargs):
    """
    Hook function for MKdocs
    """
    main()


def main() -> None:
    """
    Main function that generates Markdown tables for MainConfig and UserConfig schemas.
    """
    catalog_schema = MainConfig.model_json_schema()
    schema_to_md(catalog_schema,
                 "docs/source/main_config_descr.md")
    tuto_schema = UserConfig.model_json_schema()
    schema_to_md(tuto_schema,
                 "docs/source/user_config_descr.md")


# Call the main function to start the generation
main()
