import polars

from autofeat.convert import IntoPaths, into_columns, into_paths
from autofeat.dataset import Dataset
from autofeat.table import Table


def from_excel(
    files: IntoPaths,
    *,
    sheet_name: str | None = None,
) -> Dataset:
    """Load from Excel files.

    :param files: Excel files to load.
    :param sheet_name: Name of the sheet to load.
    :return: Dataset.
    """
    tables = []

    for path in into_paths(files):
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
