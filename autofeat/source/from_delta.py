from collections.abc import Iterable

import polars

from autofeat.convert import IntoPath, into_columns, into_path
from autofeat.dataset import Dataset
from autofeat.table import Table


def from_delta(
    files: Iterable[IntoPath],
) -> Dataset:
    """Load from Delta files.

    :param files: Delta files to load.
    :return: Dataset.
    """
    tables = []

    for file in files:
        path = into_path(file)

        data = polars.scan_delta(
            source=str(path),
        )

        table = Table(
            columns=into_columns(data),
            data=data,
            name=path.name,
        )

        tables.append(table)

    return Dataset(tables)
