import dataclasses
import itertools
from typing import Iterable

import polars

from autofeat.table import Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Combine(Transform):
    """Combine numeric columns using arithmetic operators."""

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            columns = []

            for x, y in itertools.combinations(self._numeric_columns(table), 2):
                columns.append(x + y)
                columns.append(x - y)
                columns.append(y - x)
                columns.append(x * y)
                columns.append(x / y)
                columns.append(y / x)

            yield table.apply(lambda df: df.with_columns(columns))

    def _numeric_columns(
        self,
        table: Table,
    ) -> list[polars.Expr]:
        return [
            polars.col(column)
            for column, data_type in table.schema.items()
            if data_type.is_numeric()
        ]
