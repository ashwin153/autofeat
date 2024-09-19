from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeAlias, Union

if TYPE_CHECKING:
    from autofeat.dataset import Dataset
    from autofeat.table import Table


IntoTables: TypeAlias = Union[
    "Table",
    "Dataset",
]


def into_tables(
    *values: IntoTables | Iterable[IntoTables],
) -> list[Table]:
    """Convert the ``values`` into a collection of tables.

    :param values: Values to convert.
    :return: Converted tables.
    """
    return list(_into_tables(*values))


def _into_tables(
    *values: IntoTables | Iterable[IntoTables],
) -> Iterable[Table]:
    from autofeat.dataset import Dataset
    from autofeat.table import Table

    for value in values:
        if isinstance(value, Table):
            yield value
        elif isinstance(value, Dataset):
            yield from value.tables
        elif isinstance(value, Iterable):
            yield from (t for v in value for t in _into_tables(v))
        else:
            raise NotImplementedError(f"`{type(value)}` cannot be converted to tables")
