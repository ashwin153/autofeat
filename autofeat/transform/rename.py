import dataclasses
from collections.abc import Iterable

from autofeat.schema import Schema
from autofeat.table import Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Rename(Transform):
    """Rename aliased columns.

    :param mapping: Mapping from old name to new name.
    """

    mapping: dict[str, str]

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            mapping = {
                old: new
                for old, new in self.mapping.items()
                if old in table.schema
                if new not in table.schema
            }

            schema = Schema({
                mapping.get(column, column): attributes
                for column, attributes in table.schema.items()
            })

            yield Table(
                data=table.data.rename(mapping),
                name=table.name,
                schema=schema,
            )
