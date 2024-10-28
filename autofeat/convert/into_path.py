from __future__ import annotations

import atexit
import pathlib
import shutil
import tempfile
from typing import TypeAlias, Union

import streamlit
import streamlit.runtime
import streamlit.runtime.uploaded_file_manager

_TMP = tempfile.mkdtemp()
atexit.register(shutil.rmtree, _TMP)


IntoPath: TypeAlias = Union[
    pathlib.Path,
    str,
    streamlit.runtime.uploaded_file_manager.UploadedFile,
]


def into_path(
    value: IntoPath,
) -> pathlib.Path:
    """Convert the ``value`` into a file system path.

    :param value: Value to convert.
    :return: Converted path.
    """
    if isinstance(value, pathlib.Path):
        return value
    elif isinstance(value, str):
        return pathlib.Path(value)
    elif isinstance(value, streamlit.runtime.uploaded_file_manager.UploadedFile):
        path: pathlib.Path = pathlib.Path(tempfile.mkdtemp(dir=_TMP)) / value.name
        path.write_bytes(value.getbuffer())
        return path
    else:
        raise NotImplementedError(f"{type(value)} is not supported")
