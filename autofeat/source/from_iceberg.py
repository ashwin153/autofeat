import os
from collections.abc import Iterable

import polars

from autofeat.convert import into_columns
from autofeat.dataset import Dataset
from autofeat.table import Table


def from_iceberg(
    files: Iterable[str],
) -> Dataset:
    """Load from Iceberg files.

    :param files: Iceberg files to load.
    :return: Dataset.
    """
    tables = []

    for file in files:
        data = polars.scan_iceberg(
            file,
        )

        table = Table(
            columns=into_columns(data),
            data=data,
            name=os.path.basename(str(file)),
        )

        tables.append(table)

    return Dataset(tables)
