import dataclasses
import itertools
from collections.abc import Iterable

import polars

from autofeat.attribute import Attribute
from autofeat.schema import Schema
from autofeat.table import Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Combine(Transform):
    """Combine numeric columns using arithmetic operators."""

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            numeric_columns = table.schema.select(
                include={Attribute.numeric},
                exclude={Attribute.primary_key, Attribute.pivotable},
            )

            if numeric_columns:
                key_columns = {
                    *table.schema.select(include={Attribute.primary_key}),
                    *table.schema.select(include={Attribute.pivotable}),
                }

                combined_columns = dict(
                    self._combinations(numeric_columns),
                )

                schema = Schema({
                    **{
                        column: table.schema[column]
                        for column in key_columns
                    },
                    **{
                        column: {Attribute.numeric}
                        for column in combined_columns
                    },
                })

                yield Table(
                    data=table.data.select(*key_columns, **combined_columns),
                    name=f"combine({table.name})",
                    schema=schema,
                )

    def _combinations(
        self,
        columns: Iterable[str],
    ) -> Iterable[tuple[str, polars.Expr]]:
        for x, y in itertools.combinations(columns, 2):
            yield f"{x} + {y}", polars.col(x) + polars.col(y)
            yield f"{x} - {y}", polars.col(x) - polars.col(y)
            yield f"{y} - {x}", polars.col(y) - polars.col(x)
            yield f"{x} * {y}", polars.col(x) * polars.col(y)
            yield f"{x} / {y}", polars.col(x) / polars.col(y)
            yield f"{y} / {x}", polars.col(y) / polars.col(x)
