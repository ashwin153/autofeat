import plotly.io
import streamlit
import streamlit_theme

from autofeat.settings import Settings

_CHART_THEMES = sorted(plotly.io.templates.keys())


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

        dark_mode = theme is not None and theme["base"] == "dark"

        return Settings(
            chart_theme=chart_theme,
            dark_mode=dark_mode,
        )
