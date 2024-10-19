from collections.abc import Callable
from typing import ParamSpec

import loguru
import streamlit

from autofeat import Dataset, source
from autofeat.transform import Cast, Encode, Union
from autofeat.ui.show_log import show_log

P = ParamSpec("P")


def upload_new_predictions(

) -> Dataset | None:
    """Load a dataset from a configurable source.

    :return: Loaded dataset.
    """
    csv_files = streamlit.file_uploader(
        accept_multiple_files=True,
        label="Upload Files",
        type="csv",
    )

    if not csv_files:
        return None

    return _clean_dataset(source.from_csv, tuple(file.name for file in csv_files))



@show_log("Loading New Predictions Dataset")
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

