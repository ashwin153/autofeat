from __future__ import annotations

import dataclasses
import functools
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
    table: Table

    def __str__(
        self,
    ) -> str:
        return self.name

    @property
    def data(
        self,
    ) -> polars.LazyFrame:
        """Lazily-loaded contents of this column.

        :return: Column data.
        """
        return self.table.data.select(self.expr)

    @property
    def expr(
        self,
    ) -> polars.Expr:
        """Convert this column to a Polars expression.

        :return: Converted expression.
        """
        return polars.col(self.name)

    @property
    def sample(
        self,
    ) -> polars.Series:
        """Eagerly-loaded sample of the column.

        :return: Column sample.
        """
        return self.table.sample.get_column(self.name)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Table:
    """A table of data.

    :param data: Lazily-loaded contents of this table.
    :param name: Name of this table.
    :param sample: Eagerly-loaded sample of the table.
    """

    data: polars.LazyFrame
    name: str
    sample: polars.DataFrame

    @functools.cached_property
    def columns(
        self,
    ) -> list[Column]:
        """Infer the columns in this table from the sample data.

        :return: Columns in this table.
        """
        return [
            Column(name=name, data_type=data_type, table=self)
            for name, data_type in self.sample.schema.items()
        ]

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

    def column(
        self,
        name: str,
    ) -> Column:
        """Get the column with the corresponding name.

        :param name: Name of the column.
        :return: Corresponding column.
        """
        for column in self.columns:
            if column.name == name:
                return column

        raise ValueError(f"column `{name}` does not exist")

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

    def select(
        self,
        *,
        exclude: list[str] | None = None,
        include: list[str] | None = None,
    ) -> Table:
        """Select a subset of columns from the table.

        :param exclude: Column names to exclude.
        :param include: Column names to include.
        :return: Projected table.
        """
        assert (
            not exclude
            or not include
            or all(column not in exclude for column in include)
        ), "`include` and `exclude` must be mutually exclusive"

        selector = (
            polars.selectors.by_name(include)
            if include
            else polars.all()
        )

        if exclude:
            selector = selector.exclude(exclude)

        return Table(
            data=self.data.select(selector),
            name=self.name,
            sample=self.sample.select(selector),
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
