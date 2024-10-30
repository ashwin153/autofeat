import dataclasses
from typing import Literal

PlotlyTemplate = Literal[
    "ggplot2",
    "gridon",
    "plotly",
    "plotly_dark",
    "plotly_white",
    "presentation",
    "seaborn",
    "simple_white",
    "xgridoff",
    "ygridoff",
]


PolarsEngine = Literal[
    "gpu",
    "in_memory",
    "streaming",
]


DisplayMode = Literal[
    "minimal",
    "standard",
]


@dataclasses.dataclass(kw_only=True)
class Settings:
    """Global configuration.

    :param dark_mode: Whether or not dark mode is enabled.
    :param display_mode: Density of display in the UI.
    :param plotly_template: Plotly template to use for charts in the UI.
    :param polars_engine: Polars computation backend.
    """

    dark_mode: bool = False
    display_mode: DisplayMode = "standard"
    plotly_template: PlotlyTemplate = "plotly"
    polars_engine: PolarsEngine = "streaming"


# global configuration
SETTINGS = Settings()
