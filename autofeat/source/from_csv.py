import os
import pathlib
from collections.abc import Iterable

import polars

from autofeat.dataset import Dataset
from autofeat.schema import Schema
from autofeat.table import Table

DEFAULT_CACHE = pathlib.Path.home() / ".cache" / "kaggle"


def from_csv(
    files: Iterable[str | pathlib.Path],
    *,
    ignore_errors: bool = False,
    null_values: list[str] | None = None,
) -> Dataset:
    """Load from CSV.

    :param files: CSV files to load.
    :param null_values: Values to interpret as null values
    :return: Dataset.
    """
    tables = []

    for file in files:
        data = polars.scan_csv(
            file,
            ignore_errors=ignore_errors,
            null_values=null_values,
        )

        table = Table(
            data=data,
            name=os.path.basename(str(file)),
            schema=Schema.infer(data),
        )

        tables.append(table)

    return Dataset(tables)
