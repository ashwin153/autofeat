import dataclasses
from collections.abc import Collection, Iterable, Mapping

from autofeat.schema import Schema
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
                dropped = (
                    self.columns.get(table.name, set())
                    if self.columns
                    else set()
                )

                schema = Schema({
                    column: attributes
                    for column, attributes in table.schema.items()
                    if column not in dropped
                })

                if schema:
                    yield Table(
                        data=table.data.select(schema.keys()),
                        name=table.name,
                        schema=schema,
                    )
