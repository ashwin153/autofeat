import dataclasses
from collections.abc import Iterable

from autofeat.table import Column, Table
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
                if any(old == column.name for column in table.columns)
                if all(new != column.name for column in table.columns)
            }

            columns = [
                Column(
                    name=mapping.get(column.name, column.name),
                    attributes=column.attributes,
                    derived_from=[(column, table)] if column.name in mapping else [],
                )
                for column in table.columns
            ]

            yield Table(
                data=table.data.rename(mapping),
                name=table.name,
                columns=columns,
            )
