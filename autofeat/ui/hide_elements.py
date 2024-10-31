import contextlib
from collections.abc import Iterator

import streamlit
import streamlit.delta_generator

from autofeat.settings import SETTINGS, DisplayMode


@contextlib.contextmanager
def hide_elements(
    *display_modes: DisplayMode,
) -> Iterator[None]:
    """Conditionally hide elements declared within this context.

    :param display_modes: Display modes in which the elements are not visible.
    """
    with (
        streamlit.delta_generator.DeltaGenerator(root_container=None)
        if SETTINGS.display_mode in display_modes
        else contextlib.nullcontext(),
    ):
        yield
