from collections.abc import Iterable

import attrs
import polars

from autofeat.table import Table
from autofeat.transform.base import Transform


@attrs.define(frozen=True, kw_only=True, slots=True)
class Shrink(Transform):
    """Shrink numeric columns to the minimal required data type."""

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            yield Table(
                name=table.name,
                columns=table.columns,
                data=table.data.select(polars.all().shrink_dtype()),
            )
