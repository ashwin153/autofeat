import polars

from autofeat.convert import IntoPaths, into_columns, into_paths
from autofeat.dataset import Dataset
from autofeat.table import Table


def from_parquet(
    files: IntoPaths,
    *,
    low_memory: bool = False,
) -> Dataset:
    """Load from Parquet files.

    :param files: Parquet files to load.
    :param low_memory: Reduce memory pressure at the expense of performance.
    :return: Dataset.
    """
    tables = []

    for path in into_paths(files):
        data = polars.scan_parquet(
            source=path,
            low_memory=low_memory,
        )

        table = Table(
            columns=into_columns(data),
            data=data,
            name=path.name,
        )

        tables.append(table)

    return Dataset(tables)
