import dataclasses
from collections.abc import Iterable

from autofeat.table import Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True)
class AnyOf(Transform):
    """Apply each of the ``transforms`` to every table.

    :param transforms: Transforms in any order.
    """

    transforms: Iterable[Transform]

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        tables = list(tables)

        for transform in self.transforms:
            yield from transform.apply(tables)
