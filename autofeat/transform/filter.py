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
    """

    as_of: datetime.datetime | None = None
    eq: dict[str, Any] | None = None

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            predicates = [
                *self._as_of_predicates(table),
                *self._eq_predicates(table),
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
            for column in table.columns:
                if isinstance(column.data_type, polars.Datetime):
                    yield column.expr < self.as_of
                elif isinstance(column.data_type, polars.Date):
                    yield column.expr < self.as_of.date()
                elif isinstance(column.data_type, polars.Time):
                    yield column.expr < self.as_of.time()

    def _eq_predicates(
        self,
        table: Table,
    ) -> Iterable[polars.Expr]:
        if self.eq:
            for column in table.columns:
                if value := self.eq.get(column.name):
                    yield column.expr.eq(value)
