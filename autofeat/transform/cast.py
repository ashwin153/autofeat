import dataclasses
from collections.abc import Iterable

import polars

from autofeat.attribute import Attribute
from autofeat.schema import Schema
from autofeat.table import Table
from autofeat.transform.base import Transform

# Number of rows to test casts on.
SAMPLE_SIZE = 10


@dataclasses.dataclass(frozen=True, kw_only=True)
class Cast(Transform):
    """Cast columns to more appropriate types.

    In particular, string columns containing date, time, or datetime values are converted to
    temporal columns. Casting enables further transformation (e.g., time columns enable rolling
    aggregation) and improves performance (e.g., categorical columns are easier to join).
    """

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        castable = [
            (
                table,
                table.schema.select(include={Attribute.textual, Attribute.not_null}),
            )
            for table in tables
        ]

        samples = polars.collect_all([
            (
                table.data
                .select(columns)
                .head(SAMPLE_SIZE)
            )
            for table, columns in castable
        ])

        for (table, columns), sample in zip(castable, samples):
            casted_columns = {
                column: cast
                for column in columns
                if (cast := self._cast(sample, column))
            }

            schema = Schema({
                **table.schema,
                **{c: a for c, (_, a) in casted_columns.items()},
            })

            yield Table(
                data=table.data.with_columns(**{c: e for c, (e, _) in casted_columns.items()}),
                name=table.name,
                schema=schema,
            )

    def _cast(
        self,
        sample: polars.DataFrame,
        column: str,
    ) -> tuple[polars.Expr, set[Attribute]] | None:
        to_date = polars.col(column).str.to_date("%Y-%m-%d")
        try:
            sample.select(to_date)
        except polars.exceptions.PolarsError:
            pass
        else:
            return to_date, {Attribute.temporal}

        to_datetime = polars.col(column).str.to_datetime()
        try:
            sample.select(to_datetime)
        except polars.exceptions.PolarsError:
            pass
        else:
            return to_datetime, {Attribute.temporal}

        to_time = polars.col(column).str.to_time()
        try:
            sample.select(to_time)
        except polars.exceptions.PolarsError:
            pass
        else:
            return to_time, {Attribute.temporal}

        return None
