import dataclasses
from typing import Iterable

from autofeat.table import Table, Set
from autofeat.transform.base import Transform

import polars


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
        for column, data_type in table.sample.schema.items():
            expr = polars.col(column)

            yield expr.count()
            yield expr.null_count()

            if data_type.is_numeric():
                yield expr.max()
                yield expr.mean()
                yield expr.median()
                yield expr.min()
                yield expr.std()
                yield expr.sum()
                yield expr.var()
