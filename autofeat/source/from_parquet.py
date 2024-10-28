from collections.abc import Iterable

import polars

from autofeat.convert import IntoPath, into_columns, into_path
from autofeat.dataset import Dataset
from autofeat.table import Table


def from_parquet(
    files: Iterable[IntoPath],
    *,
    low_memory: bool = False,
) -> Dataset:
    """Load from Parquet files.

    :param files: Parquet files to load.
    :param low_memory: Reduce memory pressure at the expense of performance.
    :return: Dataset.
    """
    tables = []

    for file in files:
        path = into_path(file)

        data = polars.scan_parquet(
            source=path,
            low_memory=low_memory,
        )

        table = Table(
            columns=into_columns(data),
            data=data,
            name=path.name,
        )

        tables.append(table)

    return Dataset(tables)
