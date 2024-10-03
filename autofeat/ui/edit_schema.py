from typing import Any, cast

import streamlit

from autofeat import Attribute, Dataset, Schema, Table


def edit_schema(
    dataset: Dataset,
) -> Dataset:
    with streamlit.expander("Edit Schema"):
        edited_schemas: list[Schema] = []

        for tab, table in zip(
            streamlit.tabs([table.name for table in dataset.tables]),
            dataset.tables,
        ):
            with tab:
                if streamlit.toggle("Redacted", key=table.name):
                    edited_schemas.append(Schema())
                else:
                    edited_rows = streamlit.data_editor(
                        _convert_schema_into_rows(table.schema),
                        hide_index=True,
                    )

                    edited_schema = _convert_rows_into_schema(edited_rows)

                    edited_schemas.append(edited_schema)

        return _apply_schema_changes(dataset, edited_schemas)


@streamlit.cache_resource(
    hash_funcs={
        Dataset: id,
        list: lambda schemas: tuple(
            (column, attribute)
            for schema in cast(list, schemas)
            for column, attributes in schema.items()
            for attribute in sorted(attribute.name for attribute in attributes)
        ),
    },
    max_entries=1,
)
def _apply_schema_changes(
    dataset: Dataset,
    edited_schemas: list[Schema],
) -> Dataset:
    edited_tables = []

    for table, edited_schema in zip(dataset.tables, edited_schemas):
        if edited_schema:
            edited_table = Table(
                data=table.data.select(edited_schema.keys()),
                name=table.name,
                schema=edited_schema,
            )

            edited_tables.append(edited_table)

    return Dataset(edited_tables)


@streamlit.cache_resource(
    hash_funcs={
        Schema: id,
    },
    max_entries=1,
)
def _convert_schema_into_rows(
    schema: Schema,
) -> list[dict[str, Any]]:
    return [
        {
            "column": column,
            "redacted": False,
            **{
                attribute.name: attribute in schema[column]
                for attribute in Attribute
            },
        }
        for column in sorted(schema)
    ]


@streamlit.cache_resource(
    hash_funcs={
        list: id,
    },
    max_entries=1,
)
def _convert_rows_into_schema(
    rows: list[dict[str, Any]],
) -> Schema:
    return Schema({
        cast(str, row["column"]): {
            attribute
            for attribute in Attribute
            if row[attribute.name]
        }
        for row in rows
        if not row["redacted"]
    })
