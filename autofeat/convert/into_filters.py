from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeAlias, Union

import polars

if TYPE_CHECKING:
    from autofeat.table import Column, Table
    from autofeat.transform.filter import Filter

IntoFilters: TypeAlias = Union[
    "Filter",
    "Table",
    "Column",
    polars.DataFrame,
    polars.Series,
    polars.LazyFrame,
]


def into_filters(
    *values: IntoFilters | Iterable[IntoFilters],
) -> list[Filter]:
    """Convert the ``values`` into a collection of filters.

    .. note::

        Implementation calls :meth:`polars.LazyFrame.collect`.

    :param values: Values to convert.
    :return: Converted filters.
    """
    return list(_into_filters(*values))


def _into_filters(
    *values: IntoFilters | Iterable[IntoFilters],
) -> Iterable[Filter]:
    from autofeat.table import Column, Table
    from autofeat.transform.filter import Filter

    for value in values:
        if isinstance(value, Filter):
            yield value
        elif isinstance(value, Table):
            yield from _into_filters(value.data)
        elif isinstance(value, Column):
            yield from _into_filters(value.data)
        elif isinstance(value, polars.DataFrame):
            yield from _into_filters(value.lazy())
        elif isinstance(value, polars.Series):
            yield from _into_filters(value.to_frame())
        elif isinstance(value, polars.LazyFrame):
            df = value.collect()

            time_column: str | None = next(
                (
                    column
                    for column, data_type in df.schema.items()
                    if isinstance(data_type, polars.Datetime)
                ),
                None,
            )

            yield from (
                Filter(
                    as_of=(
                        None
                        if time_column is None
                        else row[time_column]
                    ),
                    eq={
                        column: value
                        for column, value in row.items()
                        if column != time_column
                    },
                )
                for row in df.rows(named=True)
            )
        elif isinstance(value, Iterable):
            yield from (t for v in value for t in _into_filters(v))
        else:
            raise NotImplementedError(f"`{type(value)}` cannot be converted to filters")
