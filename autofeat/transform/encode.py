import dataclasses
from collections.abc import Iterable

import polars

from autofeat.table import Table
from autofeat.transform import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Encode(Transform):
    """Encode categorical variables as dummy variables using one-hot encoding."""

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            dummy_variables = []

            for column in table.columns:
                if isinstance(column.data_type, polars.Categorical):
                    categories = table.sample.select(column.expr.unique()).to_series()
                    for category in categories:
                        dummy_variables.append(
                            (column.expr == category)
                            .alias(f"{column} == {category}"),
                        )

            if dummy_variables:
                yield table.apply(lambda df: df.with_columns(dummy_variables))
            else:
                yield table
