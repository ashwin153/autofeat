from typing import Any, cast

import streamlit
import streamlit.runtime.caching.hashing

from autofeat import Attribute, Dataset, Schema, Table


def schema_editor(
    dataset: Dataset,
) -> Dataset:
    """Modify the schema of the ``dataset``.

    :param dataset: Dataset to edit.
    :return: Edited dataset.
    """
    streamlit.title("Edit Schemas")

    edits = []

    for tab, table in zip(
        streamlit.tabs([table.name for table in dataset.tables]),
        dataset.tables,
    ):
        with tab:
            rows = _schema_into_rows(table.schema)
            edited_rows = streamlit.data_editor(rows)
            edits.append(edited_rows)

    return _edit_dataset(dataset, edits)


@streamlit.cache_resource(
    hash_funcs={
        Dataset: id,
        list: lambda edits: tuple(tuple(row.values()) for edit in edits for row in edit),
    },
)
def _edit_dataset(
    dataset: Dataset,
    edits: list[list[dict[str, Any]]],
) -> Dataset:
    edited_tables = []

    for table, edited_rows in zip(dataset.tables, edits):
        if edited_schema := _schema_from_rows(edited_rows):
            edited_table = Table(
                data=table.data.select(edited_schema.keys()),
                name=table.name,
                schema=edited_schema,
            )

            edited_tables.append(edited_table)

    return Dataset(edited_tables)


def _schema_into_rows(
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


def _schema_from_rows(
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
