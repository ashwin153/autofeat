from __future__ import annotations

from typing import TypeAlias, Union

import polars

from autofeat.table import Table

IntoDataFrame: TypeAlias = Union[
    polars.DataFrame,
    polars.LazyFrame,
    "Table",
]


def into_data_frame(
    value: IntoDataFrame,
) -> polars.DataFrame:
    """Convert the ``value`` into a :class:`polars.DataFrame`.

    :param value: Value to convert.
    :return: Converted data frame.
    """
    if isinstance(value, polars.DataFrame):
        return value
    elif isinstance(value, polars.LazyFrame):
        return value.collect()
    elif isinstance(value, Table):
        return value.data.collect()
    else:
        raise NotImplementedError(f"`{type(value)}` cannot be converted to a data frame")
