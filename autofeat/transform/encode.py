from collections.abc import Iterable, Iterator

import attrs
import polars

from autofeat.attribute import Attribute
from autofeat.convert import into_exprs, into_named_exprs
from autofeat.table import Column, Table
from autofeat.transform import Transform


@attrs.define(frozen=True, kw_only=True, slots=True)
class Encode(Transform):
    """Encode categorical variables as dummy variables using one-hot encoding."""

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        tables = list(tables)

        category_iterator = iter(
            polars.collect_all([
                table.data.select(column.expr.drop_nulls().unique())
                for table in tables
                for column in self._categorical_columns(table)
            ]),
        )

        for table in tables:
            encodings = [
                *self._encodings(table, category_iterator),
            ]

            extra_columns = [
                column
                for column in table.columns
                if all(column.name != encoded_column.name for encoded_column, _ in encodings)
            ]

            columns = [
                *extra_columns,
                *[column for column, _ in encodings],
            ]

            if columns:
                yield Table(
                    data=table.data.select(
                        *into_exprs(extra_columns),
                        **into_named_exprs(encodings),
                    ),
                    name=table.name,
                    columns=columns,
                )
            else:
                yield table

    def _categorical_columns(
        self,
        table: Table,
    ) -> list[Column]:
        return [
            column
            for column in table.columns
            if Attribute.categorical in column.attributes
        ]

    def _encodings(
        self,
        table: Table,
        category_iterator: Iterator[polars.DataFrame],
    ) -> Iterable[tuple[Column, polars.Expr]]:
        for column in self._categorical_columns(table):
            categories = next(category_iterator).to_series()

            # if Attribute.textual in column.attributes:
            #     column = Column(
            #         name=column.name,
            #         attributes=column.attributes,
            #         derived_from=[(column, table)],
            #     )

            #     yield column, column.expr.cast(polars.Enum(categories=categories))

            for category in categories:
                encoded_column = Column(
                    name=f"{column} == {category}",
                    attributes={Attribute.boolean, Attribute.categorical, Attribute.not_null},
                    derived_from=[(column, table)],
                )

                yield encoded_column, column.expr == category
