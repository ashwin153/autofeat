from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, Union

import numpy
import pandas
import polars

from autofeat.settings import SETTINGS

if TYPE_CHECKING:
    from autofeat.dataset import Dataset
    from autofeat.table import Table

IntoDataFrame: TypeAlias = Union[
    numpy.ndarray,
    pandas.DataFrame,
    polars.DataFrame,
    polars.LazyFrame,
    "Table",
    "Dataset",
]


def into_data_frame(
    value: IntoDataFrame,
) -> polars.DataFrame:
    """Convert the ``value`` into a :class:`polars.DataFrame`.

    :param value: Value to convert.
    :return: Converted data frame.
    """
    from autofeat.dataset import Dataset
    from autofeat.table import Table

    if isinstance(value, numpy.ndarray):
        return polars.from_numpy(value)
    elif isinstance(value, pandas.DataFrame):
        return polars.from_pandas(value)
    elif isinstance(value, polars.DataFrame):
        return value
    elif isinstance(value, polars.LazyFrame):
        return value.collect()
    elif isinstance(value, Table):
        return value.data.collect()
    elif isinstance(value, Dataset):
        data = [
            table.data
            for table in value.tables
        ]

        match SETTINGS.polars_engine:
            case SETTINGS.PolarsEngine.gpu:
                return (
                    polars
                    .concat(data, how="horizontal")
                    .collect(engine="gpu")
                )
            case SETTINGS.PolarsEngine.in_memory:
                return polars.concat(
                    polars.collect_all(data, streaming=False),
                    how="horizontal",
                )
            case SETTINGS.PolarsEngine.streaming:
                return polars.concat(
                    polars.collect_all(data, streaming=True),
                    how="horizontal",
                )
            case _:
                raise NotImplementedError(f"f{SETTINGS.polars_engine} is not supported")
    else:
        raise NotImplementedError(f"`{type(value)}` cannot be converted to a data frame")
