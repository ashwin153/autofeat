import itertools
from collections.abc import Iterable

import attrs

from autofeat.attribute import Attribute
from autofeat.table import Table
from autofeat.transform.base import Transform


@attrs.define(frozen=True, kw_only=True, slots=True)
class Join(Transform):
    """Join tables on common primary key columns."""

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        tables = list(tables)

        primary_keys = {
            table.name: {
                column.name
                for column in table.columns
                if Attribute.primary_key in column.attributes
            }
            for table in tables
        }

        column_names = {
            table.name: {
                column.name
                for column in table.columns
            }
            for table in tables
        }

        for x, y in itertools.combinations(tables, 2):
            # TODO: should check if the primary keys share common ancestors
            if x.name not in y.name and y.name not in x.name:
                x_primary_key = primary_keys[x.name]
                x_column_names = column_names[x.name]
                y_primary_key = primary_keys[y.name]
                y_column_names = column_names[y.name]

                if y_primary_key and y_primary_key <= x_column_names:
                    columns = [
                        *list(x.columns),
                        *[column for column in y.columns if column.name not in x_column_names],
                    ]

                    yield Table(
                        data=x.data.join(y.data, on=list(y_primary_key), how="left"),
                        name=f"join({x.name}, {y.name})",
                        columns=columns,
                    )
                elif x_primary_key and x_primary_key <= y_column_names:
                    columns = [
                        *list(y.columns),
                        *[column for column in x.columns if column.name not in y_column_names],
                    ]

                    yield Table(
                        data=y.data.join(x.data, on=list(x_primary_key), how="left"),
                        name=f"join({y.name}, {x.name})",
                        columns=columns,
                    )
