import functools
from collections.abc import Iterable
from typing import TypeAlias

import polars

from autofeat.table import Table
from autofeat.transform.filter import Filter

IntoFilters: TypeAlias = (
    Filter
    | polars.LazyFrame
    | polars.DataFrame
    | Table
)


def extract_filters(
    *values: IntoFilters | Iterable[IntoFilters],
) -> list[Filter]:
    """Convert the values into a collection of filters.

    .. note::

        Implementation calls :meth:`polars.LazyFrame.collect`.

    :param values: Values to convert.
    :return: Converted filters.
    """
    filters = []

    for value_or_iterable in values:
        if isinstance(value_or_iterable, IntoFilters):
            filters.extend(_extract_filters(value_or_iterable))
        else:
            for value in value_or_iterable:
                filters.extend(_extract_filters(value))

    return filters


@functools.singledispatch
def _extract_filters(value: IntoFilters) -> list[Filter]:
    raise NotImplementedError(f"`{type(value)}` cannot be converted to a filter")


@_extract_filters.register
def _(value: Filter) -> list[Filter]:
    return [value]


@_extract_filters.register
def _(value: polars.DataFrame) -> list[Filter]:
    return _extract_filters(value.lazy())


@_extract_filters.register
def _(value: Table) -> list[Filter]:
    return _extract_filters(value.data)


@_extract_filters.register
def _(value: polars.LazyFrame) -> list[Filter]:
    df = value.collect()

    time_column: str | None = next(
        (
            column
            for column, data_type in df.schema.items()
            if isinstance(data_type, polars.Datetime)
        ),
        None,
    )

    return [
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
    ]
