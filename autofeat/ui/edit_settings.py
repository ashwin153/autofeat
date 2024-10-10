import dataclasses

import plotly.io
import streamlit
import streamlit_theme

_CHART_THEMES = sorted(plotly.io.templates.keys())


@dataclasses.dataclass(frozen=True, kw_only=True)
class Settings:
    """Application settings.

    :param chart_theme: Plotly template to use in charts.
    :param dark_mode: Whether or not dark mode is enabled.
    """

    chart_theme: str
    dark_mode: bool


def edit_settings() -> Settings:
    """Configure application settings.

    :return: Application settings.
    """
    theme = streamlit_theme.st_theme()

    with streamlit.sidebar:
        chart_theme = streamlit.selectbox(
            "Chart Theme",
            _CHART_THEMES,
            index=_CHART_THEMES.index("streamlit"),
        )

        dark_mode = theme is None or theme["base"] == "light"

        return Settings(
            chart_theme=chart_theme,
            dark_mode=dark_mode,
        )
