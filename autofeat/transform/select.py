import dataclasses
from collections.abc import Iterable

from autofeat.schema import Schema
from autofeat.table import Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Select(Transform):
    """Select a subset of columns from tables.

    :param include: Column names to include.
    :param exclude: Column names to exclude.
    """

    include: set[str] | None = None
    exclude: set[str] | None = None

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            schema = Schema({
                column: attributes
                for column, attributes in table.schema.items()
                if self.include is None or column in self.include
                if self.exclude is None or column not in self.exclude
            })

            yield Table(
                data=table.data.select(schema.keys()),
                name=table.name,
                schema=schema,
            )
