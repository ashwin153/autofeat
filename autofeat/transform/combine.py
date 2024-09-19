import dataclasses
import itertools
from collections.abc import Iterable

from autofeat.table import Column, Table
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
                columns.append((x.expr + y.expr).alias(f"{x} + {y}"))
                columns.append((x.expr - y.expr).alias(f"{x} - {y}"))
                columns.append((y.expr - x.expr).alias(f"{y} - {x}"))
                columns.append((x.expr * y.expr).alias(f"{x} * {y}"))
                columns.append((x.expr / y.expr).alias(f"{x} / {y}"))
                columns.append((y.expr / x.expr).alias(f"{y} / {x}"))

            if columns:
                yield table.apply(lambda df: df.with_columns(columns))
            else:
                yield table

    def _numeric_columns(
        self,
        table: Table,
    ) -> list[Column]:
        return [
            column
            for column in table.columns
            if column.data_type.is_numeric()
        ]
