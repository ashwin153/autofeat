from typing import cast

import pandas
import pygwalker.api.streamlit
import streamlit
import streamlit.runtime.caching.hashing

from autofeat import Attribute, Dataset, Schema, Table, source
from autofeat.transform import Aggregate, Cast, Encode, Identity, Transform

streamlit.set_page_config(
    page_title="autofeat",
    layout="wide",
)


def dataset_loader(

) -> Dataset | None:
    """Load a dataset from a configurable source.

    :return: Loaded dataset.
    """
    streamlit.title("Load Dataset")

    source_type = streamlit.selectbox(
        "Source Type",
        ["kaggle", "csv"],
    )

    if source_type == "csv":
        files = streamlit.file_uploader(
            "Upload Files",
            accept_multiple_files=True,
        )

        if files:
            return _load_dataset_from_csv([file.name for file in files])

    if source_type == "kaggle":
        name = streamlit.text_input(
            "Dataset or Competition",
            placeholder="house-prices-advanced-regression-techniques",
        )

        if name:
            return _load_dataset_from_kaggle(name)

    return None


@streamlit.cache_resource
def _load_dataset_from_csv(
    files: list[str],
) -> Dataset:
    return source.from_csv(files)


@streamlit.cache_resource
def _load_dataset_from_kaggle(
    name: str,
) -> Dataset:
    return source.from_kaggle(name)


def dataset_explorer(
    dataset: Dataset,
) -> None:
    """Explore a sample of the ``dataset``.

    :param dataset: Dataset to explore.
    """
    streamlit.title("Explore Dataset")

    sample_size = streamlit.number_input(
        "Sample Size",
        min_value=1,
        max_value=10000,
        value=100,
    )

    for tab, sample in zip(
        streamlit.tabs([table.name for table in dataset.tables]),
        _load_samples(dataset, sample_size),
    ):
        with tab:
            renderer = pygwalker.api.streamlit.StreamlitRenderer(sample)
            renderer.explorer()


@streamlit.cache_data(hash_funcs={Dataset: id})
def _load_samples(
    dataset: Dataset,
    sample_size: int,
) -> list[pandas.DataFrame]:
    return [
        table.data.head(sample_size).collect().to_pandas()
        for table in dataset.tables
    ]


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


def transform_editor(

) -> Transform:
    """Configure the transform used by feature selection.

    :return: Transform.
    """
    # todo: make this configurable
    return (
        Cast()
        .then(Encode())
        .then(Identity(), Aggregate())
    )


if original_dataset := dataset_loader():
    dataset_explorer(original_dataset)
    edited_dataset = schema_editor(original_dataset)
    transform = transform_editor()
