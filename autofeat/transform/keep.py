import dataclasses
from collections.abc import Collection, Iterable, Mapping

from autofeat.table import Column, Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Keep(Transform):
    """Keep only the ``columns`` in all tables.

    :param columns: Names of columns to keep.
    :param tables: Names of tables to keep.
    """

    columns: Mapping[str, Collection[str | Column]] | None = None
    tables: set[str] | None = None

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            if not self.tables or table.name in self.tables:
                kept = {
                    str(column)
                    for column in (self.columns.get(table.name, []) if self.columns else [])
                }

                columns = [
                    column
                    for column in table.columns
                    if column.name in kept
                ]

                if columns:
                    yield table.select(columns)
