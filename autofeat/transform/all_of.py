import dataclasses
from typing import Iterable

from autofeat.table import Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True)
class AllOf(Transform):
    """Apply all of the ``transforms`` to every table.

    :param transforms: Transforms in the order they are applied.
    """

    transforms: Iterable[Transform]

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for transform in self.transforms:
            tables = transform.apply(tables)

        yield from tables
