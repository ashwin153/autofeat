import polars

from autofeat.convert import IntoPaths, into_columns, into_paths
from autofeat.dataset import Dataset
from autofeat.table import Table


def from_delta(
    files: IntoPaths,
) -> Dataset:
    """Load from Delta files.

    :param files: Delta files to load.
    :return: Dataset.
    """
    tables = []

    for path in into_paths(files):
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
