from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, cast

import polars

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclasses.dataclass(frozen=True, kw_only=True)
class Column:
    """A column in a table.

    :param data_type: Type of data stored in the column.
    :param name: Name of the column.
    """

    data_type: polars.DataType
    name: str

    def __str__(
        self,
    ) -> str:
        return self.name

    @property
    def expr(
        self,
    ) -> polars.Expr:
        """Convert this column to a Polars expression.

        :return: Converted expression.
        """
        return polars.col(self.name)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Table:
    """A table of data.

    :param data: Lazily-loaded contents of this table.
    :param name: Name of this table.
    :param sample: Eagerly-loaded sample of the ``data``.
    """

    data: polars.LazyFrame
    name: str
    sample: polars.DataFrame

    @property
    def columns(
        self,
    ) -> set[Column]:
        """Infer the columns in this table from the sample data.

        :return: Columns in this table.
        """
        return {
            Column(name=name, data_type=data_type)
            for name, data_type in self.sample.schema.items()
        }

    def apply(
        self,
        f: Callable[
            [polars.LazyFrame | polars.DataFrame],
            polars.LazyFrame | polars.DataFrame,
        ],
        /,
    ) -> Table:
        """Apply ``f`` to this table.

        :param f: Transformation to apply.
        :return: Transformed table.
        """
        return Table(
            data=cast(polars.LazyFrame, f(self.data)),
            name=self.name,
            sample=cast(polars.DataFrame, f(self.sample)),
        )

    def is_valid(
        self,
        expr: polars.Expr,
    ) -> bool:
        """Check if the ``expr`` can be evaluated on the :attr:`~Table.sample`.

        :param expr: Expression to validate.
        """
        try:
            self.sample.select(expr)
        except polars.exceptions.PolarsError:
            return False
        else:
            return True

    def join(
        self,
        other: Table,
        /,
        on: list[str] | None = None,
        how: polars.JoinStrategy = "outer",
    ) -> Table:
        """Join this table with the ``other`` table.

        :param other: Table to join with.
        :param on: Columns to join on.
        :param how: Method of joining the tables.
        :return: Joined table.
        """
        return Table(
            data=self.data.join(other.data, on=on, how=how),
            name=f"{self.name} {how} {other.name}",
            sample=self.sample.join(other.sample, on=on, how=how),
        )

    @staticmethod
    def empty() -> Table:
        """Construct an empty table.

        :return: Empty table.
        """
        return Table(
            data=polars.LazyFrame(),
            name="empty",
            sample=polars.DataFrame(),
        )
