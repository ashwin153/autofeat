from __future__ import annotations

from typing import TypeAlias, Union

import polars

IntoSeries: TypeAlias = Union[
    polars.DataFrame,
    polars.LazyFrame,
    polars.Series,
]


def into_series(
    value: IntoSeries,
) -> polars.Series:
    """Convert the ``value`` into a :class:`polars.Series`.

    :param value: Value to convert.
    :return: Converted series.
    """
    if isinstance(value, polars.DataFrame):
        return value.to_series()
    elif isinstance(value, polars.LazyFrame):
        return value.collect().to_series()
    elif isinstance(value, polars.Series):
        return value
    else:
        raise NotImplementedError(f"`{type(value)}` cannot be converted to a series")
