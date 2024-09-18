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
    tables = []

    for value_or_iterable in values:
        if isinstance(value_or_iterable, IntoTables):
            tables.extend(_extract_tables(value_or_iterable))
        else:
            for value in value_or_iterable:
                tables.extend(_extract_tables(value))

    return tables


def _extract_tables(value: IntoTables) -> list[Table]:
    if isinstance(value, Table):
        return [value]
    elif isinstance(value, Dataset):
        return list(value.tables())
    else:
        raise NotImplementedError(f"`{type(value)}` cannot be converted to a table")
