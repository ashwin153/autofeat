import os
import pathlib
from collections.abc import Iterable
from typing import IO

import polars

from autofeat.convert import into_columns
from autofeat.dataset import Dataset
from autofeat.table import Table


def from_parquet(
    files: Iterable[str | pathlib.Path | IO[bytes]],
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
        data = polars.scan_parquet(
            file,
            low_memory=low_memory,
        )

        table = Table(
            columns=into_columns(data),
            data=data,
            name=os.path.basename(str(file)),
        )

        tables.append(table)

    return Dataset(tables)
