from typing import Any, cast

import streamlit

from autofeat import Attribute, Column, Dataset


def edit_dataset(
    dataset: Dataset,
) -> Dataset:
    with streamlit.expander("Edit Dataset"):
        edited_schemas: list[list[Column]] = []

        for tab, table in zip(
            streamlit.tabs([table.name for table in dataset.tables]),
            dataset.tables,
        ):
            with tab:
                if streamlit.toggle("Redacted", key=table.name):
                    edited_schemas.append([])
                else:
                    values = [
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

                    edited_values: list[dict[str, Any]] = streamlit.data_editor(
                        data=values,
                        hide_index=True,
                    )

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
                        for i, value in enumerate(edited_values)
                        if not value["redacted"]
                    ]

                    edited_schemas.append(edited_columns)

        return _apply_schema_changes(dataset, edited_schemas)


@streamlit.cache_resource(
    hash_funcs={
        Dataset: id,
        list: lambda schemas: tuple(
            (column.name, attribute)
            for columns in cast(list, schemas)
            for column in columns
            for attribute in sorted(attribute.name for attribute in column.attributes)
        ),
    },
    max_entries=1,
)
def _apply_schema_changes(
    dataset: Dataset,
    edited_schemas: list[list[Column]],
) -> Dataset:
    edited_tables = [
        table.select(edited_columns)
        for table, edited_columns in zip(dataset.tables, edited_schemas)
        if edited_columns
    ]

    return Dataset(edited_tables)
