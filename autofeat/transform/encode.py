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
        all_tables = [
            (
                table,
                table.schema.select(include={Attribute.categorical}),
            )
            for table in tables
        ]

        all_categories = iter(
            polars.collect_all([
                table.data.select(polars.col(column).drop_nulls().unique())
                for table, categorical_columns in all_tables
                for column in categorical_columns
            ]),
        )

        for table, categorical_columns in all_tables:
            columns = {}
            schema = Schema()

            for column, attributes in categorical_columns.items():
                categories = next(all_categories).to_series()

                if Attribute.textual in attributes:
                    columns[column] = polars.col(column).cast(polars.Enum(categories=categories))
                    schema[column] = attributes | {Attribute.pivotable}

                for category in categories:
                    dummy_variable = f"{column} == {category}"
                    columns[dummy_variable] = polars.col(column) == category
                    schema[dummy_variable] = {Attribute.boolean, Attribute.not_null}

            if columns:
                yield Table(
                    data=table.data.with_columns(**columns),
                    name=table.name,
                    schema=table.schema | schema,
                )
            else:
                yield table
