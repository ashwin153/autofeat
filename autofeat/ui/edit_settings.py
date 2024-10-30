from typing import get_args

import streamlit
import streamlit_theme

from autofeat.settings import SETTINGS, DisplayMode, PlotlyTemplate, PolarsEngine


def edit_settings() -> None:
    """Configure global settings."""
    with streamlit.sidebar:
        SETTINGS.dark_mode = "dark" == (streamlit_theme.st_theme() or {}).get("base")

        SETTINGS.display_mode = streamlit.selectbox(
            label="Display Mode",
            options=get_args(DisplayMode),
            index=get_args(DisplayMode).index("standard"),
        )

        SETTINGS.low_memory = streamlit.toggle(
            label="Low Memory",
            help="Whether or not to conserve memory at the expense of performance",
            value=False,
        )

        SETTINGS.plotly_template = streamlit.selectbox(
            label="Plotly Template",
            options=get_args(PlotlyTemplate),
            index=get_args(PlotlyTemplate).index("plotly"),
        )

        SETTINGS.polars_engine = streamlit.selectbox(
            label="Polars Engine",
            options=get_args(PolarsEngine),
            index=get_args(PolarsEngine).index("streaming"),
        )
