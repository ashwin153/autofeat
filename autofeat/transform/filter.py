from collections.abc import Iterable

import attrs

from autofeat.attribute import Attribute
from autofeat.table import Table
from autofeat.transform.base import Transform


@attrs.define(frozen=True, kw_only=True, slots=True)
class Filter(Transform):
    """Filter by categorical columns."""

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            boolean_columns = [
                column
                for column in table.columns
                if Attribute.boolean in column.attributes
                if Attribute.not_null in column.attributes
            ]

            for x in boolean_columns:
                predicates = [
                    (f"{x}", x.expr),
                    (f"not({x})", x.expr.not_()),
                ]

                for name, expr in predicates:
                    yield Table(
                        name=f"filter({table.name}, {name})",
                        columns=table.columns,
                        data=table.data.filter(expr),
                    )
