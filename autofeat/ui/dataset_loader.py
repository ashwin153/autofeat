from typing import Any, cast

import streamlit
import streamlit.runtime.caching.hashing

from autofeat import Attribute, Dataset, Schema, Table, source


def dataset_loader(

) -> Dataset | None:
    """Load a dataset from a configurable source.

    :return: Loaded dataset.
    """
    streamlit.header("Load Dataset")

    source_type = streamlit.selectbox(
        "Source Type",
        ["Kaggle", "CSV"],
    )

    if source_type == "CSV":
        csv_files = streamlit.file_uploader(
            "Upload Files",
            accept_multiple_files=True,
        )

        if not csv_files:
            return None

        dataset = _load_dataset_from_csv([file.name for file in csv_files])
    elif source_type == "Kaggle":
        kaggle_name = streamlit.text_input(
            "Dataset / Competition",
            placeholder="house-prices-advanced-regression-techniques",
        )

        if not kaggle_name:
            return None

        dataset = _load_dataset_from_kaggle(kaggle_name)
    else:
        raise NotImplementedError(f"{source_type} is not supported")

    with streamlit.expander("Edit Schema"):
        edited_schemas: list[Schema] = []

        for tab, table in zip(
            streamlit.tabs([table.name for table in dataset.tables]),
            dataset.tables,
        ):
            with tab:
                edited_rows = streamlit.data_editor(
                    _schema_into_rows(table.schema),
                    hide_index=True,
                )

                edited_schema = _schema_from_rows(edited_rows)

                edited_schemas.append(edited_schema)

        return _edit_dataset(dataset, edited_schemas)


@streamlit.cache_resource(
    max_entries=1,
)
def _load_dataset_from_csv(
    files: list[str],
) -> Dataset:
    return source.from_csv(files)


@streamlit.cache_resource(
    max_entries=1,
)
def _load_dataset_from_kaggle(
    name: str,
) -> Dataset:
    return source.from_kaggle(name)


@streamlit.cache_resource(
    hash_funcs={
        Dataset: id,
        list: lambda schemas: tuple(
            (column, attribute)
            for schema in cast(list, schemas)
            for column, attributes in schema.items()
            for attribute in sorted(attributes)
        ),
    },
    max_entries=1,
)
def _edit_dataset(
    dataset: Dataset,
    edited_schemas: list[Schema],
) -> Dataset:
    edited_tables = []

    for table, edited_schema in zip(dataset.tables, edited_schemas):
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
