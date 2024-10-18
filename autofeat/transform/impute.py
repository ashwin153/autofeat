import dataclasses
from collections.abc import Iterable

import polars

from autofeat.attribute import Attribute
from autofeat.convert.into_named_exprs import into_named_exprs
from autofeat.table import Column, Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Impute(Transform):
    """Zero-fill missing data."""

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            imputations = [
                self._impute(table, column)
                for column in table.columns
            ]

            yield Table(
                data=table.data.select(**into_named_exprs(imputations)),
                name=table.name,
                columns=[column for column, _ in imputations],
            )

    def _impute(
        self,
        table: Table,
        column: Column,
    ) -> tuple[Column, polars.Expr]:
        if Attribute.numeric in column.attributes:
            result = Column(
                name=column.name,
                attributes=column.attributes | {Attribute.not_null},
                derived_from=[(column, table)],
            )

            return result, column.expr.fill_null(value=0)

        if Attribute.boolean in column.attributes:
            result = Column(
                name=column.name,
                attributes=column.attributes | {Attribute.not_null},
                derived_from=[(column, table)],
            )

            return result, column.expr.fill_null(value=False)

        return column, column.expr
