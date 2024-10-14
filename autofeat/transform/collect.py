from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import polars

from autofeat.table import Table
from autofeat.transform import Transform

if TYPE_CHECKING:
    from collections.abc import Iterable


@attrs.define(frozen=True, kw_only=True, slots=True)
class Collect(Transform):
    """Collect all tables in memory.

    :param streaming: Whether or not to use the Polars streaming engine.
    """

    streaming: bool = True

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        tables = list(tables)

        data = polars.collect_all(
            (table.data for table in tables),
            streaming=self.streaming,
        )

        for table, df in zip(tables, data):
            yield Table(
                columns=table.columns,
                data=df.lazy(),
                name=table.name,
            )
