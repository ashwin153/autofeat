from collections.abc import Iterable

from autofeat.convert.into_tables import IntoTables, into_tables
from autofeat.dataset import Dataset


def from_tables(
    *tables: IntoTables | Iterable[IntoTables],
) -> Dataset:
    """Concatenate the tables into a dataset.

    :param tables: Tables to concatenate.
    :return: Dataset.
    """
    return Dataset(into_tables(*tables))
