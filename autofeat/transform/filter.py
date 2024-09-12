import dataclasses
import datetime
from typing import Any, Iterable

import polars

from autofeat.table import Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True)
class Filter(Transform):
    """Filter out rows that do not satisfy the predicates.

    :param as_of: Temporal filter.
    :param eq: Equality filters by column.
    :param gt: Greater than filters by column.
    :param lt: Less than filters by column.
    """

    as_of: datetime.datetime | None = None
    eq: dict[str, Any] | None = None
    gt: dict[str, Any] | None = None
    lt: dict[str, Any] | None = None

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            predicates = [
                *self._as_of_predicates(table),
                *self._eq_predicates(table),
                *self._gt_predicates(table),
                *self._lt_predicates(table),
            ]

            if predicates:
                yield table.apply(lambda df: df.filter(predicates))
            else:
                yield table

    def _as_of_predicates(
        self,
        table: Table,
    ) -> Iterable[polars.Expr]:
        if self.as_of:
            for column, data_type in table.schema.items():
                expr = polars.col(column)

                if isinstance(data_type, polars.Datetime):
                    yield expr < self.as_of
                elif isinstance(data_type, polars.Date):
                    yield expr < self.as_of.date()
                elif isinstance(data_type, polars.Time):
                    yield expr < self.as_of.time()

    def _eq_predicates(
        self,
        table: Table,
    ) -> Iterable[polars.Expr]:
        if self.eq:
            for column, value in self.eq.items():
                expr = polars.col(column)

                if column in table.schema:
                    yield expr.eq(value)

    def _gt_predicates(
        self,
        table: Table,
    ) -> Iterable[polars.Expr]:
        if self.gt:
            for column, value in self.gt.items():
                expr = polars.col(column)

                if column in table.schema:
                    yield expr.gt(value)

    def _lt_predicates(
        self,
        table: Table,
    ) -> Iterable[polars.Expr]:
        if self.lt:
            for column, value in self.lt.items():
                expr = polars.col(column)

                if column in table.schema:
                    yield expr.lt(value)
