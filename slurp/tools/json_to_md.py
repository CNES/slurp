"""
Generate Markdown tables for Tools and Tutorial schemas.
"""
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import mkdocs.plugins

from pydantic_class import MainConfig, UserConfig


def sort_fields(schema: dict[str, Any], excluded_fields: list[str]) -> tuple[
    list[str], list[str], list[str]]:
    """
    Sorts the fields of a schema into mandatory, recommended, and optional categories.

    Parameters
    ----------
    schema : dict[str, Any]
        A dictionary representing the schema containing field properties.
    excluded_fields : list[str]
        List of field names that should be excluded from sorting.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        A tuple containing three lists:
        - mandatory_fields: Fields without a default value.
        - recommended_fields: Fields marked as "recommended" in the schema.
        - optional_fields: Fields with a default value or marked as optional.
    """
    mandatory_fields = []
    recommended_fields = []
    optional_fields = []
    for field_name, meta in schema["properties"].items():
        if field_name in excluded_fields:
            continue
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


def resolve_ref(ref: str, schema: dict[str, Any]) -> dict[str, Any]:
    """
    Resolves a JSON reference ('$ref') in the schema.

    Parameters
    ----------
    ref : str
        The '$ref' string to resolve (e.g., '#/definitions/SomeModel').
    schema : dict[str, Any]
        The schema dictionary that contains the definitions.

    Returns
    -------
    dict[str, Any]
        The schema corresponding to the resolved reference.
    """
    ref_path = ref.split("/")[1:]

    resolved = schema
    for part in ref_path:
        resolved = resolved[part]
    return resolved


def parse_field_metadata(field_name: str, field_status: str, field_meta: dict[str, Any],
                         schema: dict[str, Any], fields_df: pd.DataFrame) -> None:
    """
    Parses metadata of a specific field from a schema and adds the relevant information to a DataFrame.

    This function handles various field types, including fields with `enum` values, lists of values,
    or complex types defined through `$ref` references in the schema. It extracts field descriptions,
    status, and selectable values, and inserts this information into the provided DataFrame.
    If the field has a reference (`$ref`), it recursively parses the referenced schema.

    Notes
    -----
    This function modifies the input DataFrame `fields_df` in place by adding new rows with field metadata.

    Parameters
    ----------
    field_name : str
        The name of the field being parsed.
    field_status : str
        The status of the field (e.g., "mandatory", "optional").
    field_meta : dict[str, Any]
        A dictionary containing metadata about the field, such as its description or possible values.
    schema : dict[str, Any]
        The overall schema where the field is defined, used to resolve references (`$ref`).
    fields_df : pd.DataFrame
        A DataFrame that is populated with the metadata of the parsed field.
        The DataFrame has columns:
        - "Metadata" : str, the field name.
        - "Description" : str, the field description.
        - "selectable values" : str, possible values if the field has an enum.
        - "Status" : str, the status of the field (e.g., "mandatory", "optional").
    """
    selectable_values = ""
    if "enum" in field_meta:
        # Literal
        selectable_values = ", ".join(field_meta["enum"])
    elif "items" in field_meta and "enum" in field_meta["items"]:
        # Optional[Literal]
        selectable_values = ", ".join(field_meta["items"]["enum"])
    elif "anyOf" in field_meta:
        for val in field_meta["anyOf"]:
            # Recommended: Optional[Literal]
            if "enum" in val:
                selectable_values = ", ".join(val["enum"])
            # Optional[list[Literal]]
            if "items" in val and "enum" in val["items"]:
                selectable_values = ", ".join(val["items"]["enum"])
            # Handling "$ref"
            if "$ref" in val:
                fields_df.loc[len(fields_df.index)] = [field_name, field_meta["description"],
                                                       selectable_values, field_status]
                # Resolve the reference and recursively get the field metadata
                referenced_meta_schema = resolve_ref(val["$ref"], schema)
                mandatory_fields, recommended_fields, optional_fields = sort_fields(
                    referenced_meta_schema, [])
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

    fields_df.loc[len(fields_df.index)] = [field_name, field_meta["description"],
                                           selectable_values, field_status]


def fields_descriptions(fields_status: list[tuple[str, list[str]]],
                        schema: dict[str, Any]) -> pd.DataFrame:
    """
    Creates a DataFrame containing field metadata descriptions and status.

    Parameters
    ----------
    fields_status : list[tuple[str, list[str]]]
        A list of tuples where each tuple contains a field status (e.g., "mandatory") and
        the corresponding list of field names.
    schema : dict[str, Any]
        A dictionary representing the schema containing metadata about each field.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - Metadata: Field name.
        - Description: Field description from the schema.
        - selectable values: If the field has "enum" values, these will be listed here.
        - Status: The status of the field (e.g., mandatory, optional).
    """
    fields_df = pd.DataFrame(
        columns=["Metadata", "Description", "selectable values", "Status"])
    for status, fields in fields_status:
        for field in fields:
            field_meta = schema["properties"][field]
            parse_field_metadata(field, status, field_meta, schema, fields_df)
    return fields_df


def schema_to_md(schema: dict[str, Any], output_md_file: Path) -> None:
    """
    Converts a schema to a Markdown table and saves it to a file.

    Parameters
    ----------
    schema : dict[str, Any]
        A dictionary representing the schema containing field properties.
    output_md_file : Path
        The output path where the Markdown file will be saved.
    """
    excluded_fields = list(RecommendedParametersValidatorMixin.model_fields.keys()) + ["flagship"]
    fields_df_desc = fields_to_df(excluded_fields, schema)
    fields_df_desc.to_markdown(buf=output_md_file, index=False)


def fields_to_df(excluded_fields, schema):
    """
    Converts a schema to a DataFrame

    Parameters
    ----------
    excluded_fields : list[Str]
        A list of fields not kept in the DataFrame
    schema : dict[str, Any]
        A dictionary representing the schema containing field properties.
    Returns
    -------
    pd.DataFrame
        A DataFrame describing the schema.
    """
    mandatory_fields, recommended_fields, optional_fields = sort_fields(schema, excluded_fields)
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

main()