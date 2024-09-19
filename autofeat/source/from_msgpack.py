import io

import ormsgpack
import polars

from autofeat.dataset import Dataset
from autofeat.table import Table


def from_msgpack(
    value: bytes,
    /,
) -> Dataset:
    """Deserialize a dataset from the Msgpack-encoded ``value``.

    :param value: Msgpack-encoded bytes.
    :return: Deserialized dataset.
    """
    tables = [
        Table(
            data=polars.LazyFrame.deserialize(io.BytesIO(table["data"])),
            name=table["name"],
            sample=polars.read_parquet(table["sample"]),
        )
        for table in ormsgpack.unpackb(value)["tables"]
    ]

    return Dataset(tables)
