import polars

from autofeat.convert import IntoPaths, into_columns, into_paths
from autofeat.dataset import Dataset
from autofeat.table import Table


def from_iceberg(
    files: IntoPaths,
) -> Dataset:
    """Load from Iceberg files.

    :param files: Iceberg files to load.
    :return: Dataset.
    """
    tables = []

    for path in into_paths(files):
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
