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
        return list(value)  # type: ignore[arg-type]
    else:
        raise NotImplementedError(f"{type(value)} is not supported")


def _infer_columns(
    data: polars.LazyFrame,
) -> list[Column]:
    from autofeat.attribute import Attribute
    from autofeat.table import Column

    schema = data.collect_schema()

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

    return [
        Column(
            name=name,
            attributes=Attribute.infer(
                data_type=data_type,
                len=profile["len"][name],
                n_unique=profile["n_unique"][name],
                null_count=profile["null_count"][name],
            ),
        )
        for name, data_type in schema.items()
    ]
