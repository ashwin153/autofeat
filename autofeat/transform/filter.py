import datetime
from collections.abc import Collection, Iterable
from typing import Any

import attrs
import polars

from autofeat.attribute import Attribute
from autofeat.table import Table
from autofeat.transform.base import Transform


@attrs.define(frozen=True, kw_only=True, slots=True)
class Filter(Transform):
    """Filter out rows that do not satisfy the predicates.

    :param as_of: Temporal constraint.
    :param eq: Equality constraints.
    :param gt: Greater-than constraints.
    :param is_in: Set constraints.
    :param lt: Less-than constraints.
    """

    as_of: datetime.datetime | None = None
    eq: dict[str, Any] | None = None
    gt: dict[str, Any] | None = None
    is_in: dict[str, Collection[Any]] | None = None
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
                *self._is_in_predicates(table),
                *self._lt_predicates(table),
            ]

            if predicates:
                yield Table(
                    columns=table.columns,
                    data=table.data.filter(predicates),
                    name=table.name,
                )

    def _as_of_predicates(
        self,
        table: Table,
    ) -> Iterable[polars.Expr]:
        if self.as_of:
            for column in table.columns:
                if Attribute.temporal in column.attributes:
                    yield column.expr < self.as_of

    def _eq_predicates(
        self,
        table: Table,
    ) -> Iterable[polars.Expr]:
        if self.eq:
            for column in table.columns:
                if value := self.eq.get(column.name):
                    yield column.expr.eq(value)

    def _gt_predicates(
        self,
        table: Table,
    ) -> Iterable[polars.Expr]:
        if self.gt:
            for column in table.columns:
                if value := self.gt.get(column.name):
                    yield column.expr.gt(value)

    def _is_in_predicates(
        self,
        table: Table,
    ) -> Iterable[polars.Expr]:
        if self.is_in:
            for column in table.columns:
                if values := self.is_in.get(column.name):
                    yield column.expr.is_in(values)

    def _lt_predicates(
        self,
        table: Table,
    ) -> Iterable[polars.Expr]:
        if self.lt:
            for column in table.columns:
                if value := self.lt.get(column.name):
                    yield column.expr.lt(value)
