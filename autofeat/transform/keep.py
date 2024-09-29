import dataclasses
from collections.abc import Iterable

from autofeat.schema import Schema
from autofeat.table import Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Keep(Transform):
    """Keep only the ``columns`` in all tables.

    :param columns: Names of columns to keep.
    :param tables: Names of tables to keep.
    """

    columns: dict[str, set[str]] | None = None
    tables: set[str] | None = None

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            if not self.tables or table.name in self.tables:
                kept = (
                    self.columns.get(table.name, set())
                    if self.columns
                    else set()
                )

                schema = Schema({
                    column: attributes
                    for column, attributes in table.schema.items()
                    if column in kept
                })

                if schema:
                    yield Table(
                        data=table.data.select(schema.keys()),
                        name=table.name,
                        schema=schema,
                    )
