from collections.abc import Iterable

from autofeat.convert.into_tables import IntoTables, into_tables
from autofeat.dataset import Dataset


def from_tables(
    *tables: IntoTables | Iterable[IntoTables],
) -> Dataset:
    """

    :param tables:
    :return:
    """
    return Dataset(into_tables(tables))
