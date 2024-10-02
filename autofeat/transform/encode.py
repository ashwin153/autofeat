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
            (table, table.schema.select(include={Attribute.categorical}))
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

            for column in categorical_columns:
                categories = next(all_categories).to_series()

                # TODO: known columns need to be cast to enum for this to work
                # if Attribute.textual in attributes:
                #     columns[column] = polars.col(column).cast(polars.Enum(categories=categories))
                #     schema[column] = attributes | {Attribute.pivotable}

                for category in categories:
                    dummy_variable = f"{column} == {category}"
                    columns[dummy_variable] = polars.col(column) == category
                    schema[dummy_variable] = {Attribute.boolean, Attribute.pivotable}

            if columns:
                yield Table(
                    data=table.data.with_columns(**columns),
                    name=f"encode({table.name})",
                    schema=table.schema | schema,
                )
            else:
                yield table
