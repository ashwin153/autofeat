import dataclasses
from typing import Any, Iterable

import polars

from autofeat.table import Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True)
class Filter(Transform):
    """Filter out rows that do not satisfy the predicates.

    :param eq: Equality filters by column.
    :param gt: Greater than filters by column.
    :param lt: Less than filters by column.
    """

    eq: dict[str, Any] | None = None
    gt: dict[str, Any] | None = None
    lt: dict[str, Any] | None = None

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            predicates = [
                *self._eq_predicates(table),
                *self._gt_predicates(table),
                *self._lt_predicates(table),
            ]

            if predicates:
                yield table.apply(lambda df: df.filter(predicates))
            else:
                yield table

    def _eq_predicates(
        self,
        table: Table,
    ) -> Iterable[polars.Expr]:
        if self.eq:
            yield from (
                polars.col(column).filter(polars.col(column).eq(value))
                for column, value in self.eq.items()
                if column in table.schema
            )

    def _gt_predicates(
        self,
        table: Table,
    ) -> Iterable[polars.Expr]:
        if self.gt:
            yield from (
                polars.col(column).filter(polars.col(column).gt(value))
                for column, value in self.gt.items()
                if column in table.schema
            )

    def _lt_predicates(
        self,
        table: Table,
    ) -> Iterable[polars.Expr]:
        if self.lt:
            yield from (
                polars.col(column).filter(polars.col(column).lt(value))
                for column, value in self.lt.items()
                if column in table.schema
            )
