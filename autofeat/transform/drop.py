from collections.abc import Collection, Iterable

import attrs

from autofeat.table import Column, Table
from autofeat.transform.base import Transform


@attrs.define(frozen=True, kw_only=True, slots=True)
class Drop(Transform):
    """Drop any of the ``columns`` from all tables.

    :param columns: Columns to drop.
    :param tables: Tables to drop.
    """

    columns: Collection[tuple[Column, Table]] | None = None
    tables: Collection[Table] | None = None

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        column_names = {
            (column.name, table.name)
            for column, table in self.columns or []
        }

        table_names = {
            table.name
            for table in self.tables or []
        }

        for table in tables:
            if self.tables and table.name in table_names:
                continue

            if self.columns:
                remaining_columns = [
                    column
                    for column in table.columns
                    if (column.name, table.name) not in column_names
                ]

                if remaining_columns:
                    table = table.select(remaining_columns)
                else:
                    continue

            yield table
