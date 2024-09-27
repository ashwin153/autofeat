import dataclasses
from collections.abc import Iterable

from autofeat.schema import Schema
from autofeat.table import Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Keep(Transform):
    """Keep only the ``columns`` in all tables.

    :param columns: Names of columns to keep.
    """

    columns: set[str]

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            schema = Schema({
                column: attributes
                for column, attributes in table.schema.items()
                if column in self.columns
            })

            yield Table(
                data=table.data.select(schema.keys()),
                name=table.name,
                schema=schema,
            )
