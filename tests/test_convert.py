import polars.testing
import pytest

from autofeat import Column, convert
from autofeat.convert import IntoNamedExprs, IntoSeries


@pytest.mark.parametrize(
    "given,expected",
    [
        (
            [Column(name="x")],
            {"x": polars.col("x")},
        ),
        (
            [Column(name="x"), (Column(name="y"), polars.col("x").mean())],
            {"x": polars.col("x"), "y": polars.col("x").mean()},
        ),
    ],
)
def test_into_named_exprs(
    given: IntoNamedExprs,
    expected: dict[str, polars.Expr],
) -> None:
    actual = convert.into_named_exprs(given)
    assert len(actual) == len(expected)
    for name, expr in actual.items():
        assert str(expr) == str(expected[name])


@pytest.mark.parametrize(
    "given,expected",
    [
        (
            polars.Series("x", [1, 2, 3]),
            polars.Series("x", [1, 2, 3]),
        ),
        (
            polars.DataFrame({"x": [1, 2, 3]}),
            polars.Series("x", [1, 2, 3]),
        ),
        (
            polars.LazyFrame({"x": [1, 2, 3]}),
            polars.Series("x", [1, 2, 3]),
        ),
    ],
)
def test_into_series(
    given: IntoSeries,
    expected: polars.Series,
) -> None:
    polars.testing.assert_series_equal(
        expected,
        convert.into_series(given),
    )
