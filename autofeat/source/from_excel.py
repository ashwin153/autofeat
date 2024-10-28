import os
import pathlib
from collections.abc import Iterable
from typing import IO

import polars

from autofeat.convert import into_columns
from autofeat.dataset import Dataset
from autofeat.table import Table


def from_excel(
    files: Iterable[str | pathlib.Path | IO[bytes]],
    *,
    sheet_name: str | None = None,
) -> Dataset:
    """Load from Excel files.

    :param files: Excel files to load.
    :param sheet_name: Name of the sheet to load.
    :return: Dataset.
    """
    tables = []

    for file in files:
        data = polars.read_excel(
            file,
            sheet_name=sheet_name,
        )

        table = Table(
            columns=into_columns(data),
            data=data.lazy(),
            name=os.path.basename(str(file)),
        )

        tables.append(table)

    return Dataset(tables)
