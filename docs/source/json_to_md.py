from typing import get_args, get_origin, List, Union
import pandas as pd
from pydantic import BaseModel, Field
from slurp.tools.pydantic_class import MainConfig


def format_type(t):
    origin = get_origin(t)

    # Union
    if origin is Union:
        args = [a for a in get_args(t) if a is not type(None)]
        if len(args) == 1:
            return f"{format_type(args[0])}"
        return " | ".join(format_type(a) for a in args)

    # List
    if origin in (list, List):
        inner = ", ".join(format_type(a) for a in get_args(t))
        return f"List[{inner}]"

    # Builtin types
    if isinstance(t, type):
        return t.__name__.capitalize()
    return str(t)


# Recursive field extractor
def extract_field_info(model_class, prefix="", indent=0):
    """Recursively extract table rows with indentation for subclasses."""
    table_data = []
    for field_name, model_field in model_class.model_fields.items():
        full_field_name = f"{prefix}.{field_name}" if prefix else field_name
        field_type = model_field.annotation
        description = model_field.description or ""
        status = "Mandatory" if model_field.is_required() else "Optional"

        display_name = f"{'&nbsp;&nbsp;' * indent}**{field_name}**" if indent == 0 else f"{'&nbsp;&nbsp;' * indent}{field_name}"

        # Get default value
        if model_field.default is not None:
            default = repr(model_field.default)
        elif model_field.default_factory is not None:
            default = "<factory>"
        elif model_field.is_required():
            default = ""
        else:
            default = "None"

        # Append default to type string
        type_str = format_type(field_type)
        if default != "PydanticUndefined":
            type_str += f" (Default {default})"

        # Handle nested models
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            table_data.append([display_name, "", "", description, status])
            table_data.extend(extract_field_info(field_type, prefix=full_field_name, indent=indent + 1))
        else:
            table_data.append(["", display_name, type_str, description, status])

    return table_data


# Generate and save Markdown table
def generate_markdown_table(config_class, output_path):
    table_data = extract_field_info(config_class)
    df = pd.DataFrame(table_data, columns=["Field Group", "Field Name", "Expected Type", "Description", "Status"])
    markdown_str = df.to_markdown(index=False)
    with open(output_path, "w") as f:
        f.write(markdown_str)
    return markdown_str


def main() -> None:
    """
    Main function that generates Markdown tables for MainConfig and UserConfig schemas.
    """
    generate_markdown_table(MainConfig, "main_config_descr.md")

