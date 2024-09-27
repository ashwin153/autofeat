import dataclasses
from collections.abc import Iterable

import polars

from autofeat.attribute import Attribute
from autofeat.schema import Schema
from autofeat.table import Table
from autofeat.transform import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Encode(Transform):
    """Encode categorical variables as dummy variables using one-hot encoding."""

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        encodable = [
            (table, categorical_columns)
            for table in tables
            if (categorical_columns := table.schema.select(include={Attribute.categorical}))
        ]

        categories = iter(
            polars.collect_all([
                table.data.select(polars.col(column).drop_nulls().unique())
                for table, categorical_columns in encodable
                for column in categorical_columns
            ]),
        )

        for table, categorical_columns in encodable:
            key_columns = {
                *table.schema.select(include={Attribute.primary_key}),
                *table.schema.select(include={Attribute.pivotable}),
            }

            dummy_columns = {
                f"{column} == {category}": polars.col(column) == category
                for column in categorical_columns
                for category in next(categories).to_series()
            }

            schema = Schema({
                **{
                    column: table.schema[column]
                    for column in key_columns
                },
                **{
                    column: {Attribute.boolean}
                    for column in dummy_columns
                },
            })

            yield Table(
                data=table.data.select(*key_columns, **dummy_columns),
                name=f"encode({table.name})",
                schema=schema,
            )
