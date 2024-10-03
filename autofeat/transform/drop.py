import dataclasses
from collections.abc import Collection, Iterable, Mapping

from autofeat.table import Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Drop(Transform):
    """Drop any of the ``columns`` from all tables.

    :param columns: Names of columns to drop.
    :param tables: Names of tables to drop.
    """

    columns: Mapping[str, Collection[str]] | None = None
    tables: set[str] | None = None

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            if not self.tables or table.name not in self.tables:
                dropped = {
                    str(column)
                    for column in (self.columns.get(table.name, []) if self.columns else [])
                }

                columns = [
                    column
                    for column in table.columns
                    if column.name not in dropped
                ]

                if columns:
                    yield table.select(columns)
