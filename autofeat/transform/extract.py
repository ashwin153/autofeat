import dataclasses
from collections.abc import Iterable
from typing import ClassVar

import polars

from autofeat.attribute import Attribute
from autofeat.convert import IntoDataFrame, into_data_frame, into_named_exprs
from autofeat.table import Column, Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Extract(Transform):
    """Extract features that are relevant to the ``known`` data.

    :param known: Data that is already known.
    """

    # Reserved characters used to separate column and table names.
    SEPARATOR: ClassVar = " :: "

    known: IntoDataFrame

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        known = into_data_frame(self.known)

        for table in tables:
            primary_key = {
                column.name
                for column in table.columns
                if Attribute.primary_key in column.attributes
            }

            features = [
                *self._features(table),
            ]

            if (
                primary_key
                and primary_key.issubset(known.columns)
                and features
            ):
                columns = [
                    column
                    for column, _ in features
                ]

                data = (
                    known
                    .lazy()
                    .join(table.data, on=list(primary_key), how="left")
                    .select(**into_named_exprs(features))
                )

                yield Table(
                    columns=columns,
                    data=data,
                    name=f"features({table.name})",
                )

    def _features(
        self,
        table: Table,
    ) -> Iterable[tuple[Column, polars.Expr]]:
        for x in table.columns:
            if  (
                {Attribute.boolean, Attribute.numeric} & x.attributes
                and Attribute.primary_key not in x.attributes
            ):
                column = Column(
                    name=f"{x.name}{Extract.SEPARATOR}{table.name}",
                    attributes=x.attributes,
                    derived_from=[(x, table)],
                )

                yield column, x.expr
