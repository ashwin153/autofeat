from typing import get_args

import streamlit
import streamlit_theme

from autofeat.settings import SETTINGS, DisplayMode, PlotlyTemplate, PolarsEngine, Settings


def edit_settings() -> None:
    """Configure global settings."""
    default_settings = Settings()

    with streamlit.sidebar:
        SETTINGS.dark_mode = streamlit.toggle(
            "Dark Mode",
            disabled=True,
            value=(
                "dark" == theme.get("base")
                if (theme := streamlit_theme.st_theme())
                else default_settings.dark_mode
            ),
        )

        SETTINGS.display_mode = streamlit.selectbox(
            label="Display Mode",
            options=get_args(DisplayMode),
            index=get_args(DisplayMode).index(default_settings.display_mode),
        )

        SETTINGS.low_memory = streamlit.toggle(
            label="Low Memory",
            help="Whether or not to conserve memory at the expense of performance",
            value=default_settings.low_memory,
        )

        SETTINGS.plotly_template = streamlit.selectbox(
            label="Plotly Template",
            options=get_args(PlotlyTemplate),
            index=get_args(PlotlyTemplate).index(default_settings.plotly_template),
        )

        SETTINGS.polars_engine = streamlit.selectbox(
            label="Polars Engine",
            options=get_args(PolarsEngine),
            index=get_args(PolarsEngine).index(default_settings.polars_engine),
        )
