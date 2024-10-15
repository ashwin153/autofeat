from collections.abc import Iterable

import attrs

from autofeat.table import Table
from autofeat.transform.base import Transform


@attrs.define(frozen=True, slots=True)
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
