from collections.abc import Callable, Iterable

import attrs

from autofeat.table import Table
from autofeat.transform.base import Transform


@attrs.define(frozen=True, slots=True)
class Require(Transform):
    """Filter out tables that do not satisfy a predicate.

    .. tip::

        Used to exclude certain tables from further transformation.
    """

    predicate: Callable[[Table], bool]

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            if self.predicate(table):
                yield table
