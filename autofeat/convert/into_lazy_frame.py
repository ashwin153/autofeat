from __future__ import annotations

from typing import TypeAlias, Union

import polars

from autofeat.table import Table

IntoLazyFrame: TypeAlias = Union[
    polars.DataFrame,
    polars.LazyFrame,
    "Table",
]


def into_lazy_frame(
    value: IntoLazyFrame,
) -> polars.LazyFrame:
    """Convert the ``value`` into a :class:`polars.LazyFrame`.

    :param value: Value to convert.
    :return: Converted lazy frame.
    """
    if isinstance(value, polars.DataFrame):
        return value.lazy()
    elif isinstance(value, polars.LazyFrame):
        return value
    elif isinstance(value, Table):
        return value.data
    else:
        raise NotImplementedError(f"`{type(value)}` cannot be converted to a lazy frame")