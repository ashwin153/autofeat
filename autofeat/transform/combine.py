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
                x_expr = polars.col(x)
                y_expr = polars.col(y)

                columns.append((x_expr + y_expr).alias(f"{x} + {y}"))
                columns.append((x_expr - y_expr).alias(f"{x} - {y}"))
                columns.append((y_expr - x_expr).alias(f"{y} - {x}"))
                columns.append((x_expr * y_expr).alias(f"{x} * {y}"))
                columns.append((x_expr / y_expr).alias(f"{x} / {y}"))
                columns.append((y_expr / x_expr).alias(f"{y} / {x}"))

            yield table.apply(lambda df: df.select(columns))

    def _numeric_columns(
        self,
        table: Table,
    ) -> list[str]:
        return [
            column
            for column, data_type in table.schema.items()
            if data_type.is_numeric()
        ]
