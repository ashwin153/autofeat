from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeAlias, Union

import polars

if TYPE_CHECKING:
    from autofeat.table import Column


_IntoExpr: TypeAlias = Union[
    "Column",
    polars.Expr,
]


IntoExprs: TypeAlias = Union[
    _IntoExpr,
    Iterable[_IntoExpr],
]


def into_exprs(
    *values: IntoExprs,
) -> list[polars.Expr]:
    """Convert the ``values`` into Polars expressions.

    :param values: Values to convert.
    :return: Converted expressions.
    """
    return list(_into_exprs(*values))


def _into_exprs(
    *values: IntoExprs,
) -> Iterable[polars.Expr]:
    from autofeat.table import Column

    for value in values:
        if isinstance(value, Column):
            yield value.expr
        elif isinstance(value, polars.Expr):
            yield value
        elif isinstance(value, Iterable):
            yield from (y for x in value for y in _into_exprs(x))
        else:
            raise NotImplementedError(f"{type(value)} is not supported")
