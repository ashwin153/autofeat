import streamlit
import streamlit_theme

from autofeat.settings import SETTINGS


def edit_settings() -> None:
    """Configure global settings."""
    with streamlit.sidebar:
        SETTINGS.plotly_template = streamlit.selectbox(
            "Plotly Template",
            SETTINGS.PlotlyTemplate,
            index=list(SETTINGS.PlotlyTemplate).index(SETTINGS.PlotlyTemplate.plotly),
        )

        SETTINGS.dark_mode = "dark" == (streamlit_theme.st_theme() or {}).get("base")

        SETTINGS.polars_engine = streamlit.selectbox(
            "Polars Engine",
            SETTINGS.PolarsEngine,
            index=list(SETTINGS.PolarsEngine).index(SETTINGS.PolarsEngine.streaming),
        )
