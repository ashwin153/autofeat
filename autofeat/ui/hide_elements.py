import contextlib
from collections.abc import Iterator

import streamlit

from autofeat.settings import SETTINGS, DisplayMode


@contextlib.contextmanager
def hide_elements(
    *display_modes: DisplayMode,
) -> Iterator[None]:
    """Conditionally hide elements declared within this context.

    :param display_modes: Display modes in which the elements are not visible.
    """
    parent = streamlit.empty()

    with parent.container():
        yield

    if SETTINGS.display_mode in display_modes:
        parent.empty()
