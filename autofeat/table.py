from __future__ import annotations

import dataclasses
from typing import Callable, Set, cast

import polars


@dataclasses.dataclass(frozen=True, kw_only=True)
class Table:
    """A table of data.

    :param data: Lazily-loaded contents of this table.
    :param sample: Eagerly-loaded sample of the ``data``.
    """

    data: polars.LazyFrame
    sample: polars.DataFrame

    @property
    def columns(
        self,
    ) -> Set[str]:
        """Infer the columns in this table from the sample.

        :return: Columns in this table.
        """
        return set(self.sample.schema)

    @property
    def schema(
        self,
    ) -> polars.Schema:
        """Infer the schema of this table from the sample.

        :return: Schema of this table.
        """
        return self.sample.schema

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
        except polars.exceptions.ComputeError:
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
        :param on: Columns to join on. (inferred)
        :param how: Method of joining the tables. (outer)
        :return: Joined table.
        """
        return Table(
            data=self.data.join(other.data, on=on, how=how),
            sample=self.sample.join(other.sample, on=on, how=how),
        )

    @staticmethod
    def empty() -> Table:
        """Construct an empty table.

        :return: Empty table.
        """
        return Table(data=polars.LazyFrame(), sample=polars.DataFrame())