from typing import Any, cast

import streamlit

from autofeat import Attribute, Column, Dataset


def edit_dataset(
    dataset: Dataset,
) -> Dataset:
    with streamlit.expander("Edit Dataset"):
        edited_schemas: list[list[dict[str, Any]]] = []

        for tab, table, schema in zip(
            streamlit.tabs([table.name for table in dataset.tables]),
            dataset.tables,
            _load_schemas(dataset),
        ):
            with tab:
                if streamlit.toggle("Redacted", key=table.name):
                    edited_schemas.append([])
                else:
                    edited_schema = streamlit.data_editor(
                        schema,
                        hide_index=True,
                        disabled=["column"],
                    )

                    edited_schemas.append(edited_schema)

        return _edit_schemas(dataset, edited_schemas)


@streamlit.cache_resource(
    hash_funcs={
        Dataset: id,
        list: lambda schemas: tuple(
            item
            for schema in cast(list, schemas)
            for value in schema
            for item in value.items()
        ),
    },
    max_entries=1,
)
def _edit_schemas(
    dataset: Dataset,
    edited_schemas: list[list[dict[str, Any]]],
) -> Dataset:
    edited_tables = []

    for table, edited_schema in zip(dataset.tables, edited_schemas):
        edited_columns = [
            Column(
                name=table.columns[i].name,
                attributes={
                    attribute
                    for attribute in Attribute
                    if value[attribute.name]
                },
                derived_from=table.columns[i].derived_from,
            )
            for i, value in enumerate(edited_schema)
            if not value["redacted"]
        ]

        if edited_columns:
            edited_tables.append(table.select(edited_columns))

    return Dataset(edited_tables)


@streamlit.cache_resource(
    hash_funcs={
        Dataset: id,
    },
    max_entries=1,
)
def _load_schemas(
    dataset: Dataset,
) -> list[list[dict[str, Any]]]:
    return [
        [
            {
                "column": column.name,
                "redacted": False,
                **{
                    attribute.name: attribute in column.attributes
                    for attribute in Attribute
                },
            }
            for column in table.columns
        ]
        for table in dataset.tables
    ]
