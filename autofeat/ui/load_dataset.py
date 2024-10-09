from collections.abc import Callable
from typing import ParamSpec

import streamlit

from autofeat import Dataset, source
from autofeat.transform import Cast, Encode

P = ParamSpec("P")


def load_dataset(

) -> Dataset | None:
    """Load a dataset from a configurable source.

    :return: Loaded dataset.
    """
    source_type = streamlit.selectbox(
        help="Location where your data is stored",
        label="Source Type",
        options=["CSV", "Kaggle", "Example"],
        index=1,
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

            return _source_dataset(source.from_csv, tuple(file.name for file in csv_files))
        case "Example":
            return _source_dataset(source.from_example)
        case "Kaggle":
            kaggle_name = streamlit.text_input(
                help="Name of the Kaggle dataset or competition to load data from",
                label="Dataset / Competition / URL",
                placeholder="house-prices-advanced-regression-techniques",
            )

            if not kaggle_name:
                return None

            return _source_dataset(source.from_kaggle, kaggle_name)
        case _:
            raise NotImplementedError(f"{source_type} is not supported")


@streamlit.cache_resource(
    max_entries=1,
    show_spinner="Loading Dataset",
)
def _source_dataset(
    _loader: Callable[P, Dataset],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Dataset:
    dataset = _loader(*args, **kwargs)
    return dataset.apply(Cast().then(Encode()))
