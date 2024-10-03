from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeAlias, Union

import polars

if TYPE_CHECKING:
    from autofeat.table import Column


_IntoNamedExprs: TypeAlias = Union[
    "Column",
    tuple["Column", polars.Expr],
]


IntoNamedExprs: TypeAlias = Iterable[_IntoNamedExprs]


def into_named_exprs(
    values: IntoNamedExprs,
) -> dict[str, polars.Expr]:
    """Convert the ``values`` into Polars named expressions.

    :param values: Values to convert.
    :return: Converted named expressions.
    """
    return dict(_into_named_exprs(values))


def _into_named_exprs(
    values: IntoNamedExprs,
) -> Iterable[tuple[str, polars.Expr]]:
    from autofeat.table import Column

    for value in values:
        if isinstance(value, Column):
            yield (value.name, polars.col(value.name))
        elif isinstance(value, tuple):
            yield (value[0].name, value[1])
        else:
            raise NotImplementedError(f"{type(value)} is not supported")
