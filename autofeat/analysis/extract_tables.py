from collections.abc import Iterable
from typing import TypeAlias

from autofeat.dataset import Dataset
from autofeat.table import Table

IntoTables: TypeAlias = (
    Table
    | Dataset
)


def extract_tables(
    *values: IntoTables | Iterable[IntoTables],
) -> list[Table]:
    """Convert the values into a collection of tables.

    :param values: Values to convert.
    :return: Converted tables.
    """
    return list(_extract_tables(*values))


def _extract_tables(
    *values: IntoTables | Iterable[IntoTables],
) -> Iterable[Table]:
    for value in values:
        if isinstance(value, Iterable):
            yield from (t for v in value for t in _extract_tables(v))
        if isinstance(value, Table):
            yield value
        elif isinstance(value, Dataset):
            yield from value.tables()
        else:
            raise NotImplementedError(f"`{type(value)}` cannot be converted to tables")
