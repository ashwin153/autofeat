from typing import cast

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

    edited_tables = []

    for tab, table in zip(
        streamlit.tabs([table.name for table in dataset.tables]),
        dataset.tables,
    ):
        with tab:
            edited_rows = streamlit.data_editor([
                {
                    "column": column,
                    "redacted": False,
                    **{
                        attribute.name: attribute in table.schema[column]
                        for attribute in Attribute
                    },
                }
                for column in sorted(table.schema)
            ])

            edited_schema = Schema({
                cast(str, row["column"]): {
                    attribute
                    for attribute in Attribute
                    if row[attribute.name]
                }
                for row in edited_rows
                if not row["redacted"]
            })

            edited_table = Table(
                data=table.data.select(edited_schema.keys()),
                name=table.name,
                schema=edited_schema,
            )

            edited_tables.append(edited_table)

    return Dataset(edited_tables)
