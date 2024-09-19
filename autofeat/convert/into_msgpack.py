import dataclasses
import io
from typing import Any, TypeAlias

import ormsgpack
import polars

IntoMsgpack: TypeAlias = Any

def into_msgpack(
    value: IntoMsgpack,
) -> bytes:
    """Serialize the ``value`` to Msgpack-encoded bytes.

    :return: Msgpack-encoded bytes.
    """
    return ormsgpack.packb(
        value,
        default=_default,
        option=ormsgpack.OPT_PASSTHROUGH_DATACLASS,
    )


def _default(
    obj: Any,
) -> Any:
    if isinstance(obj, polars.LazyFrame):
        return obj.serialize()
    elif isinstance(obj, polars.DataFrame):
        buffer = io.BytesIO()
        obj.write_parquet(buffer)
        buffer.seek(0)
        return buffer.getvalue()
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    else:
        raise TypeError(f"cannot serialize {type(obj)} to msgpack")
