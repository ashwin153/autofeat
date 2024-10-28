from collections.abc import Iterable

import polars

from autofeat.convert import IntoPath, into_columns, into_path
from autofeat.dataset import Dataset
from autofeat.table import Table


def from_iceberg(
    files: Iterable[IntoPath],
) -> Dataset:
    """Load from Iceberg files.

    :param files: Iceberg files to load.
    :return: Dataset.
    """
    tables = []

    for file in files:
        path = into_path(file)

        data = polars.scan_iceberg(
            source=path,
        )

        table = Table(
            columns=into_columns(data),
            data=data,
            name=path.name,
        )

        tables.append(table)

    return Dataset(tables)
