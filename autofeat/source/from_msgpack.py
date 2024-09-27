import io

import ormsgpack
import polars

from autofeat.dataset import Dataset
from autofeat.schema import Schema
from autofeat.table import Table


def from_msgpack(
    value: bytes,
    /,
) -> Dataset:
    """Load from Msgpack.

    :param value: Msgpack-encoded bytes.
    :return: Dataset.
    """
    tables = [
        Table(
            data=polars.LazyFrame.deserialize(io.BytesIO(table["data"])),
            name=table["name"],
            schema=Schema(table["schema"]),
        )
        for table in ormsgpack.unpackb(value)["tables"]
    ]

    return Dataset(tables)
