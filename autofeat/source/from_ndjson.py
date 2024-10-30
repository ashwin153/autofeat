from collections.abc import Iterable

import polars

from autofeat.convert import IntoPath, into_columns, into_path
from autofeat.dataset import Dataset
from autofeat.settings import SETTINGS
from autofeat.table import Table


def from_ndjson(
    files: Iterable[IntoPath],
) -> Dataset:
    """Load from newline-delimited JSON files.

    :param files: Newline-delimited JSON files to load.
    :return: Dataset.
    """
    tables = []

    for file in files:
        path = into_path(file)

        data = polars.scan_ndjson(
            source=path,
            low_memory=SETTINGS.low_memory,
        )

        table = Table(
            columns=into_columns(data),
            data=data,
            name=path.name,
        )

        tables.append(table)

    return Dataset(tables)
