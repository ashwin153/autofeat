import dataclasses
from typing import Iterable

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

            yield table.apply(lambda df: df.rename(mapping))
