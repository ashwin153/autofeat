from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, Union

import polars

if TYPE_CHECKING:
    from autofeat.dataset import Dataset
    from autofeat.table import Table

IntoDataFrame: TypeAlias = Union[
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

    if isinstance(value, polars.DataFrame):
        return value
    elif isinstance(value, polars.LazyFrame):
        return value.collect()
    elif isinstance(value, Table):
        return value.data.collect()
    elif isinstance(value, Dataset):
        return polars.concat(
            polars.collect_all([table.data for table in value.tables], streaming=True),
            how="horizontal",
        )
    else:
        raise NotImplementedError(f"`{type(value)}` cannot be converted to a data frame")
