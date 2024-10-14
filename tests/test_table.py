import polars
import pytest

from autofeat import Attribute, Column, Table

X = Column(
    name="x",
    attributes={Attribute.numeric},
)


Y = Column(
    name="y",
    attributes={Attribute.boolean},
)

TABLE = Table(
    name="example",
    data=polars.LazyFrame({X.name: [1, 2, 3], Y.name: [True, False, True]}),
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


def test_select() -> None:
    result = TABLE.select([X])
    assert result.name == TABLE.name
    assert result.columns == [X]
    polars.testing.assert_frame_equal(result.data.collect(), TABLE.data.select(X.name).collect())
