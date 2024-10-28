import os
import pathlib
from collections.abc import Iterable

import polars

from autofeat.convert import into_columns
from autofeat.dataset import Dataset
from autofeat.table import Table


def from_delta(
    files: Iterable[str | pathlib.Path],
) -> Dataset:
    """Load from Delta files.

    :param files: Delta files to load.
    :return: Dataset.
    """
    tables = []

    for file in files:
        data = polars.scan_delta(
            str(file),
        )

        table = Table(
            columns=into_columns(data),
            data=data,
            name=os.path.basename(str(file)),
        )

        tables.append(table)

    return Dataset(tables)
