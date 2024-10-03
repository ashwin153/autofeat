from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeAlias, Union

import polars

if TYPE_CHECKING:
    from autofeat.table import Column

IntoColumns: TypeAlias = Union[
    polars.LazyFrame,
    polars.DataFrame,
    "Column",
    Iterable["Column"],
]


def into_columns(
    value: IntoColumns,
) -> list[Column]:
    """Convert the ``value`` to columns using schema inference.

    .. note::

        Schema inference is a computationally expensive operation.

    :param value: Value to convert to columns.
    :return: Converted columns.
    """
    from autofeat.table import Column

    if isinstance(value, Column):
        return [value]
    elif isinstance(value, polars.LazyFrame):
        return _infer_columns(value)
    elif isinstance(value, polars.DataFrame):
        return _infer_columns(value.lazy())
    elif isinstance(value, Iterable):
        return list(value)
    else:
        raise NotImplementedError(f"{type(value)} is not supported")


def _infer_columns(
    data: polars.LazyFrame,
) -> list[Column]:
    from autofeat.attribute import Attribute
    from autofeat.table import Column

    # profile the data
    metrics = {
        "len":
            data.select(polars.all().len()),
        "n_unique":
            data.select(polars.all().n_unique()),
        "null_count":
            data.select(polars.all().null_count()),
    }

    profile = {
        metric: df.row(0, named=True)
        for metric, df in zip(
            metrics.keys(),
            polars.collect_all(metrics.values()),
        )
    }

    # use the profile and the schema of the data to infer column attributes
    columns = []

    for column_name, data_type in data.collect_schema().items():
        attributes = set()

        if isinstance(data_type, polars.Boolean):
            attributes.add(Attribute.boolean)

        if profile["n_unique"][column_name] <= 50:
            attributes.add(Attribute.categorical)

        if profile["null_count"][column_name] == 0:
            attributes.add(Attribute.not_null)

        if data_type.is_numeric():
            attributes.add(Attribute.numeric)

        if (
            profile["n_unique"][column_name] < profile["len"][column_name] * 0.10
            and (data_type.is_integer() or isinstance(data_type, polars.String))
        ):
            attributes.add(Attribute.pivotable)

        if profile["n_unique"][column_name] == profile["len"][column_name]:
            attributes.add(Attribute.primary_key)

        if isinstance(data_type, polars.String):
            attributes.add(Attribute.textual)

        columns.append(Column(name=column_name, attributes=attributes))

    return columns
