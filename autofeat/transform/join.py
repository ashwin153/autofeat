import dataclasses
import itertools
from collections.abc import Iterable

import polars
import polars._typing

from autofeat.attribute import Attribute
from autofeat.table import Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Join(Transform):
    """Join tables on common primary key columns.

    :param how: Method of joining the tables.
    """

    how: polars._typing.JoinStrategy = "inner"

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        primary_keys = {
            table.name: set(table.schema.select(include={Attribute.primary_key}))
            for table in tables
        }

        for x, y in itertools.combinations(tables, 2):
            if x.name not in y.name and y.name not in x.name:
                x_primary_key = primary_keys[x.name]
                y_primary_key = primary_keys[y.name]

                if y_primary_key and y_primary_key <= x_primary_key:
                    yield Table(
                        data=x.data.join(y.data, on=list(y_primary_key), how=self.how),
                        name=f"{self.how}_join({x.name}, {y.name})",
                        schema=y.schema | x.schema,
                    )
                elif x_primary_key and x_primary_key <= y_primary_key:
                    yield Table(
                        data=y.data.join(x.data, on=list(x_primary_key), how=self.how),
                        name=f"{self.how}_join({y.name}, {x.name})",
                        schema=x.schema | y.schema,
                    )
