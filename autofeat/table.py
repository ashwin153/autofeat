from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING

import polars

if TYPE_CHECKING:

    from autofeat.attribute import Attribute


@dataclasses.dataclass(frozen=True, kw_only=True)
class Column:
    """A column in a table.

    :param attributes: Metadata associated with this column.
    :param derived_from: Columns that this column was derived from.
    :param name: Unique name of the column within the table.
    """

    attributes: set[Attribute] = dataclasses.field(default_factory=set, repr=False)
    derived_from: list[tuple[Column, Table]] = dataclasses.field(default_factory=list, repr=False)
    name: str

    def __str__(
        self,
    ) -> str:
        return self.name

    @functools.cached_property
    def expr(
        self,
    ) -> polars.Expr:
        """Convert this column to a Polars expression.

        :return: Polars expression.
        """
        return polars.col(self.name)

    def is_related(
        self,
        other: Column,
        /,
    ) -> bool:
        """Whether or not the columns are derived from a common ancestor.

        :param other: Other column.
        :return: Has common ancestor.
        """
        return not self._ancestors.isdisjoint(other._ancestors)

    @functools.cached_property
    def _ancestors(
        self,
    ) -> set[str]:
        if self.derived_from:
            return {
                ancestor
                for parent, _ in self.derived_from
                for ancestor in parent._ancestors
            }
        else:
            return {self.name}


@dataclasses.dataclass(frozen=True, kw_only=True)
class Table:
    """A lazily-loaded data table.

    :param columns: Columns in this table.
    :param data: Contents of this table.
    :param name: Name of this table.
    """

    columns: list[Column] = dataclasses.field(compare=False, repr=False)
    data: polars.LazyFrame = dataclasses.field(compare=False, repr=False)
    name: str

    def __str__(
        self,
    ) -> str:
        return self.name

    def column(
        self,
        name: str,
    ) -> Column:
        """Get the column with the corresponding ``name``.

        :param name: Name of the column.
        :return: Corresponding column.
        """
        for column in self.columns:
            if column.name == name:
                return column

        raise ValueError(f"column {name} does not exist")

    def select(
        self,
        columns: list[Column],
    ) -> Table:
        """Project the ``columns`` from this table.

        :return: Projected table.
        """
        return Table(
            columns=columns,
            data=self.data.select(column.name for column in columns),
            name=self.name,
        )
