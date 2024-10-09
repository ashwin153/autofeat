import polars.testing
import pytest

from autofeat import convert


@pytest.mark.parametrize(
    "given",
    [
        polars.Series("x", [1, 2, 3]),
        polars.DataFrame({"x": [1, 2, 3]}),
        polars.LazyFrame({"x": [1, 2, 3]}),
    ],
)
def test_into_series(given) -> None:
    polars.testing.assert_series_equal(
        polars.Series("x", [1, 2, 3]),
        convert.into_series(given),
    )
