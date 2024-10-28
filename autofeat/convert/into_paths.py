from __future__ import annotations

import atexit
import pathlib
import shutil
import tempfile
from collections.abc import Iterable
from typing import TypeAlias, Union

import streamlit
import streamlit.runtime
import streamlit.runtime.uploaded_file_manager

_TMP = tempfile.mkdtemp()
atexit.register(shutil.rmtree, _TMP)


_IntoPath: TypeAlias = Union[
    pathlib.Path,
    str,
    streamlit.runtime.uploaded_file_manager.UploadedFile,
]


IntoPaths: TypeAlias = Union[
    _IntoPath,
    Iterable[_IntoPath],
]


def into_paths(
    *values: IntoPaths,
) -> list[pathlib.Path]:
    """Convert the ``values`` into paths.

    :param values: Values to convert.
    :return: Converted paths.
    """
    return list(_into_paths(*values))


def _into_paths(
    *values: IntoPaths,
) -> Iterable[pathlib.Path]:
    for value in values:
        if isinstance(value, pathlib.Path):
            yield value
        elif isinstance(value, str):
            yield pathlib.Path(value)
        elif isinstance(value, streamlit.runtime.uploaded_file_manager.UploadedFile):
            path = pathlib.Path(tempfile.mkdtemp(dir=_TMP)) / value.name
            path.write_bytes(value.getbuffer())
            yield path
        else:
            raise NotImplementedError(f"{type(value)} is not supported")
