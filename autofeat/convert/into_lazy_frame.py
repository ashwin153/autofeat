from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, Union

import numpy
import pandas
import polars

if TYPE_CHECKING:
    from autofeat.dataset import Dataset
    from autofeat.table import Table


IntoLazyFrame: TypeAlias = Union[
    numpy.ndarray,
    pandas.DataFrame,
    polars.DataFrame,
    polars.LazyFrame,
    "Table",
    "Dataset",
]


def into_lazy_frame(
    value: IntoLazyFrame,
) -> polars.LazyFrame:
    """Convert the ``value`` into a :class:`polars.LazyFrame`.

    :param value: Value to convert.
    :return: Converted lazy frame.
    """
    from autofeat.dataset import Dataset
    from autofeat.table import Table

    if isinstance(value, numpy.ndarray):
        return polars.from_numpy(value).lazy()
    elif isinstance(value, pandas.DataFrame):
        return polars.from_pandas(value).lazy()
    elif isinstance(value, polars.DataFrame):
        return value.lazy()
    elif isinstance(value, polars.LazyFrame):
        return value
    elif isinstance(value, Table):
        return value.data
    elif isinstance(value, Dataset):
        return polars.concat(
            [table.data for table in value.tables],
            how="horizontal",
        )
    else:
        raise NotImplementedError(f"`{type(value)}` cannot be converted to a lazy frame")
