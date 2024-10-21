from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, Union

import numpy
import pandas
import polars

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
        # TODO: explore using the gpu
        # return (
        #     polars
        #     .concat([table.data for table in value.tables], how="horizontal")
        #     .collect(engine="gpu")
        # )
        # TODO: put streaming back here
        return polars.concat(
            polars.collect_all([table.data for table in value.tables]),
            how="horizontal",
        )
    else:
        raise NotImplementedError(f"`{type(value)}` cannot be converted to a data frame")
