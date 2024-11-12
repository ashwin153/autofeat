from collections.abc import Iterable

import polars

from autofeat.convert import IntoPath, into_columns, into_path
from autofeat.dataset import Dataset
from autofeat.settings import SETTINGS
from autofeat.table import Table


def from_csv(
    files: Iterable[IntoPath],
    *,
    ignore_errors: bool = True,
    null_values: list[str] | None = None,
) -> Dataset:
    """Load from CSV files.

    :param files: CSV files to load.
    :param ignore_errors: Keep reading even if some lines are invalid.
    :param null_values: Values to interpret as null values.
    :return: Dataset.
    """
    tables = []

    for file in files:
        path = into_path(file)

        data = polars.scan_csv(
            ignore_errors=ignore_errors,
            low_memory=SETTINGS.low_memory,
            null_values=null_values,
            source=path,
        )

        table = Table(
            columns=into_columns(data),
            data=data,
            name=path.name,
        )

        tables.append(table)

    return Dataset(tables)
