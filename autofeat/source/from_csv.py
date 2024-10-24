import os
import pathlib
from collections.abc import Iterable
from typing import IO

import polars

from autofeat.convert import into_columns
from autofeat.dataset import Dataset
from autofeat.table import Table


def from_csv(
    files: Iterable[str | pathlib.Path | IO[str] | IO[bytes]],
    *,
    ignore_errors: bool = False,
    low_memory: bool = False,
    null_values: list[str] | None = None,
) -> Dataset:
    """Load from CSV.

    :param files: CSV files to load.
    :param ignore_errors: Keep reading even if some lines are invalid.
    :param low_memory: Reduce memory pressure at the expense of performance.
    :param null_values: Values to interpret as null values.
    :return: CSV dataset.
    """
    tables = []

    for file in files:
        data = polars.scan_csv(
            file,
            ignore_errors=ignore_errors,
            low_memory=low_memory,
            null_values=null_values,
        )

        table = Table(
            columns=into_columns(data),
            data=data,
            name=os.path.basename(str(file)),
        )

        tables.append(table)

    return Dataset(tables)
