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
        files = streamlit.file_uploader(
            "Upload Files",
            accept_multiple_files=True,
        )

        if not files:
            return None

        dataset = _load_dataset_from_csv([file.name for file in files])
    elif source_type == "Kaggle":
        name = streamlit.text_input(
            "Dataset / Competition",
            placeholder="house-prices-advanced-regression-techniques",
        )

        if not name:
            return None

        dataset = _load_dataset_from_kaggle(name)
    else:
        raise NotImplementedError(f"{source_type} is not supported")

    with streamlit.expander("Edit Schema"):
        edits = []

        for tab, table in zip(
            streamlit.tabs([table.name for table in dataset.tables]),
            dataset.tables,
        ):
            with tab:
                old = _schema_into_rows(table.schema)
                new = streamlit.data_editor(old)

                if old == new:
                    edits.append([])
                else:
                    edits.append(new)

        return _edit_dataset(dataset, edits)


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
        list: lambda edits: tuple(
            tuple(row.values())
            for edit in cast(list, edits)
            for row in edit
        ),
    },
    max_entries=1,
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
        else:
            edited_tables.append(table)

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
