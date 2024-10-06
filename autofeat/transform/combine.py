import dataclasses
import itertools
from collections.abc import Iterable

import polars

from autofeat.attribute import Attribute
from autofeat.convert import into_exprs, into_named_exprs
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
            if combinations := list(self._combinations(table)):
                extra_columns = [
                    column
                    for column in table.columns
                    if Attribute.primary_key in column.attributes
                ]

                columns = [
                    *extra_columns,
                    *[column for column, _ in combinations],
                ]

                yield Table(
                    data=table.data.select(*into_exprs(extra_columns), **into_named_exprs(columns)),
                    name=f"combine({table.name})",
                    columns=columns,
                )

    def _combinations(
        self,
        table: Table,
    ) -> Iterable[tuple[Column, polars.Expr]]:
        numeric_columns = [
            column
            for column in table.columns
            if Attribute.numeric in column.attributes
            if Attribute.primary_key not in column.attributes
        ]

        for x, y in itertools.combinations(numeric_columns, 2):
            combinations = [
                (f"{x} + {y}", x.expr + y.expr),
                (f"{x} - {y}", x.expr - y.expr),
                (f"{y} - {x}", y.expr - x.expr),
                (f"{x} * {y}", x.expr * y.expr),
                (f"{x} / {y}", x.expr / y.expr),
                (f"{y} / {x}", y.expr / x.expr),
            ]

            for name, expr in combinations:
                column = Column(
                    name=name,
                    attributes=x.attributes & y.attributes,
                    derived_from=[(x, table), (y, table)],
                )

                yield column, expr

        boolean_columns = [
            column
            for column in table.columns
            if Attribute.boolean in column.attributes
            if Attribute.primary_key not in column.attributes
        ]

        for x, y in itertools.combinations(boolean_columns, 2):
            combinations = [
                (f"{x} & {y}", x.expr & y.expr),
                (f"{x} | {y}", x.expr | y.expr),
            ]

            for name, expr in combinations:
                column = Column(
                    name=name,
                    attributes=x.attributes & y.attributes,
                    derived_from=[(x, table), (y, table)],
                )

                yield column, expr
