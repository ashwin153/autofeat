import polars
import pytest

from autofeat import Column, Table

X = Column(
    name="x",
)


Y = Column(
    name="y",
)

TABLE = Table(
    name="example",
    data=polars.LazyFrame({X.name: [1, 2, 3]}),
    columns=[X, Y],
)

X_MIN = Column(
    name=f"min({X})",
    derived_from=[(X, TABLE)],
)

X_MAX = Column(
    name=f"max({X})",
    derived_from=[(X, TABLE)],
)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (X, X, True),
        (X, X_MIN, True),
        (X, X_MAX, True),
        (X, Y, False),
        (X_MIN, X, True),
        (X_MIN, X_MIN, True),
        (X_MIN, X_MAX, True),
        (X_MIN, Y, False),
        (X_MAX, X, True),
        (X_MAX, X_MIN, True),
        (X_MAX, X_MAX, True),
        (X_MAX, Y, False),
        (Y, X, False),
        (Y, X_MIN, False),
        (Y, X_MAX, False),
        (Y, Y, True),
    ],
)
def test_is_related(
    *,
    a: Column,
    b: Column,
    expected: bool,
) -> None:
    assert a.is_related(b) == expected
