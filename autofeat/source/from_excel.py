from collections.abc import Iterable

import polars

from autofeat.convert import IntoPath, into_columns, into_path
from autofeat.dataset import Dataset
from autofeat.table import Table


def from_excel(
    files: Iterable[IntoPath],
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
        path = into_path(file)

        data = polars.read_excel(
            source=path,
            sheet_name=sheet_name,
        )

        table = Table(
            columns=into_columns(data),
            data=data.lazy(),
            name=path.name,
        )

        tables.append(table)

    return Dataset(tables)
