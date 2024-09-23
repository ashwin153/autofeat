import dataclasses
import itertools
from collections.abc import Iterable

import polars

from autofeat.table import Column, Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Combine_And_Or(Transform):
    """Combine boolean columns using logical AND and OR operators."""

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            columns = []

            for x, y in itertools.combinations(self._boolean_columns(table), 2):
                columns.append((x.expr & y.expr).alias(f"{x} AND {y}"))
                columns.append((x.expr | y.expr).alias(f"{x} OR {y}"))

            if columns:
                yield table.apply(lambda df: df.with_columns(columns))
            else:
                yield table

    def _boolean_columns(
        self,
        table: Table,
    ) -> list[Column]:
        return [
            column
            for column in table.columns
            if column.data_type == polars.Boolean
        ]
