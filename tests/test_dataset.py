import pickle

import polars
import pytest

from autofeat import Column, Dataset, Table

DATASET = Dataset(
    tables=[
        Table(
            name="example",
            data=polars.LazyFrame({"x": [1, 2, 3]}),
            columns=[Column(name="x")],
        ),
    ],
)


def test_is_picklable() -> None:
    # streamlit disk caching uses pickle
    pickle.dumps(DATASET)


def test_table() -> None:
    assert DATASET.table("example")

    with pytest.raises(ValueError):
        DATASET.table("unknown")
