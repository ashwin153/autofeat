import dataclasses
from collections.abc import Iterable

from autofeat.table import Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Select(Transform):
    """Select a subset of columns from tables.

    :param include: Column names to include.
    :param exclude: Column names to exclude.
    """

    include: list[str] | None = None
    exclude: list[str] | None = None

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            yield table.select(include=self.include, exclude=self.exclude)
