import contextlib
import logging
from collections.abc import Iterator

import loguru
import streamlit


@contextlib.contextmanager
def show_log(
    label: str,
) -> Iterator[None]:
    """Display logs emitted during a long-running operation.

    :param label: Short description of the operation.
    :return: Context manager.
    """
    with streamlit.status(label):
        with streamlit.empty():
            handler_id = loguru.logger.add(_ShowLog())
            try:
                yield None
            finally:
                loguru.logger.remove(handler_id)


class _ShowLog(logging.Handler):
    def __init__(
        self,
        messages: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._messages = messages or []

    def emit(
        self,
        record: logging.LogRecord,
    ) -> None:
        self._messages.append(record.msg)
        streamlit.code("\n".join(self._messages), language="log")
