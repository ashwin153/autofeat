import datetime

import polars.testing
import pytest

from autofeat import Table, convert
from autofeat.transform import AllOf, AnyOf, Cast, Combine, Encode, Identity, Transform


@pytest.mark.parametrize(
    "transform,given,expected",
    [
        (
            Cast(),
            [polars.LazyFrame({"date": ["2021-02-03"]})],
            [polars.DataFrame({"date": [datetime.date(2021, 2, 3)]})],
        ),
        (
            Cast(),
            [polars.LazyFrame({"datetime": ["2021-02-03T04:05:06"]})],
            [polars.DataFrame({"datetime": [datetime.datetime(2021, 2, 3, 4, 5, 6)]})],
        ),
        (
            Cast(),
            [polars.LazyFrame({"time": ["09:30:05"]})],
            [polars.DataFrame({"time": [datetime.time(9, 30, 5)]})],
        ),
    ],
)
def test_apply(
    transform: Transform,
    given: list[polars.LazyFrame],
    expected: list[polars.DataFrame],
) -> None:
    tables = [
        Table(data=df, name=str(i), schema=convert.into_columns(df))
        for i, df in enumerate(given)
    ]

    actual = [
        table.data.collect()
        for table in transform.apply(tables)
    ]

    assert len(actual) == len(expected)

    for x, y in zip(actual, expected):
        polars.testing.assert_frame_equal(x, y)


@pytest.mark.parametrize(
    "actual,expected",
    [
        (
            Identity().then(Identity()),
            Identity(),
        ),
        (
            Identity().then(Cast()),
            Cast(),
        ),
        (
            Cast().then(Identity()),
            Cast(),
        ),
        (
            Cast().then(Combine()).then(Encode()),
            AllOf([Cast(), Combine(), Encode()]),
        ),
        (
            Cast().then(Combine(), Identity()),
            AnyOf([AllOf([Cast(), Combine()]), Cast()]),
        ),
    ],
)
def test_then(
    actual: Transform,
    expected: Transform,
) -> None:
    assert actual == expected
