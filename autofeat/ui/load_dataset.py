from collections.abc import Callable
from typing import ParamSpec

import loguru
import streamlit

from autofeat import Dataset, source
from autofeat.transform import Cast, Encode, Union
from autofeat.ui.show_log import show_log

P = ParamSpec("P")


_SOURCE_TYPES = [
    "BigQuery (coming soon)",
    "CSV",
    "Example",
    "Iceburg (coming soon)",
    "Kaggle",
    "Parquet (coming soon)",
    "RedShift (coming soon)",
    "Salesforce (coming soon)",
    "Snowflake (coming soon)",
]


def load_dataset(

) -> Dataset | None:
    """Load a dataset from a configurable source.

    :return: Loaded dataset.
    """
    source_type = streamlit.selectbox(
        help="Location where your data is stored",
        label="Source Type",
        options=_SOURCE_TYPES,
        index=_SOURCE_TYPES.index("Kaggle"),
    )

    match source_type:
        case "CSV":
            csv_files = streamlit.file_uploader(
                accept_multiple_files=True,
                label="Upload Files",
                type="csv",
            )

            if not csv_files:
                return None

            return _clean_dataset(source.from_csv, tuple(file.name for file in csv_files))
        case "Example":
            return _clean_dataset(source.from_example)
        case "Kaggle":
            kaggle_name = streamlit.text_input(
                help="Name of the Kaggle dataset or competition to load data from",
                label="Dataset / Competition / URL",
                placeholder="house-prices-advanced-regression-techniques",
            )

            if not kaggle_name:
                return None

            return _clean_dataset(source.from_kaggle, kaggle_name)
        case _:
            raise NotImplementedError(f"{source_type} is not supported")


@show_log("Loading Dataset")
@streamlit.cache_resource(
    max_entries=1,
    show_spinner=False,
)
def _clean_dataset(
    _loader: Callable[P, Dataset],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Dataset:
    loguru.logger.info("scanning dataset")
    dataset = _loader(*args, **kwargs)

    loguru.logger.info("casting data types")
    dataset = dataset.apply(Cast())

    loguru.logger.info("concatenating tables")
    dataset = dataset.apply(Union())

    loguru.logger.info("encoding categorical variables")
    dataset = dataset.apply(Encode())

    return dataset
