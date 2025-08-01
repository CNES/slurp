from pathlib import Path
from typing import Any, List, Dict, Tuple, Type, Union, Optional
import pandas as pd
from pydantic import BaseModel, Field
import mkdocs.plugins
from pydantic_class import MainConfig, UserConfig


def extract_field_info(field_type: Type[BaseModel], prefix: str = '') -> List[List[str]]:
    """
    Recursively extracts field information from Pydantic models (including nested models) and
    returns it as a list of metadata rows for Markdown tables.
    """
    table_data = []

    # Check if the field is a Pydantic model
    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        for field_name, field_info in field_type.__annotations__.items():
            description = field_info.__doc__ if field_info.__doc__ else "No description"
            full_field_name = f"{prefix}.{field_name}" if prefix else field_name
            table_data.append([full_field_name, str(field_type.__name__), description])
            # Recursively extract nested field info
            table_data.extend(extract_field_info(field_info, prefix=full_field_name))

    return table_data


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

    # Now, check if it's a Pydantic model (directly or via List, Union, Optional)
    field_type = field_meta.get("type")
    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        # Direct subclass of BaseModel
        fields_df.loc[len(fields_df.index)] = [field_name,
                                               f"{field_meta['description']} (Pydantic class: {field_type.__name__})",
                                               selectable_values, field_status]
        fields_df = parse_nested_fields(field_name, field_type, fields_df)

    elif hasattr(field_type, "__origin__") and field_type.__origin__ in [list, List, Union, Optional]:
        # Check for List[...] or Union[...] or Optional[...] containing Pydantic subclasses
        for arg in field_type.__args__:
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                fields_df.loc[len(fields_df.index)] = [field_name,
                                                       f"{field_meta['description']} (List/Union/Optional containing {arg.__name__})",
                                                       selectable_values, field_status]
                fields_df = parse_nested_fields(field_name, arg, fields_df)

    else:
        # Otherwise, just add the field to the table
        fields_df.loc[len(fields_df.index)] = [field_name, field_meta["description"],
                                               selectable_values, field_status]


def parse_nested_fields(parent_field: str, nested_field_type: Type[BaseModel], fields_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse nested fields from a Pydantic model and add them to the DataFrame.
    """
    nested_schema = nested_field_type.schema()
    for nested_field, nested_meta in nested_schema["properties"].items():
        full_field_name = f"{parent_field}.{nested_field}"
        parse_field_metadata(full_field_name, "nested", nested_meta, nested_schema, fields_df)
    return fields_df


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
