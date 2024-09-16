import dataclasses
from typing import Iterable

import polars

from autofeat.table import Set, Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Aggregate(Transform):
    """Group by a set of columns and aggregate the remaining columns in various ways.

    :param by: Columns to group by.
    """

    by: Set[str]

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            if aggregations := list(self._aggregations(table)):
                if by := self.by & table.columns:
                    yield table.apply(lambda df: df.group_by(by).agg(*aggregations))
                else:
                    yield table.apply(lambda df: df.select(aggregations))

    def _aggregations(
        self,
        table: Table,
    ) -> Iterable[polars.Expr]:
        yield polars.count().alias("count(*)")

        for column, data_type in table.sample.schema.items():
            expr = polars.col(column)

            if data_type.is_numeric():
                yield expr.max().alias(f"max({column})")
                yield expr.mean().alias(f"mean({column})")
                yield expr.median().alias(f"median({column})")
                yield expr.min().alias(f"min({column})")
                yield expr.std().alias(f"std({column})")
                yield expr.sum().alias(f"sum({column})")
                yield expr.var().alias(f"var({column})")
