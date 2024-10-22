import dataclasses
import enum


@dataclasses.dataclass(kw_only=True)
class Settings:
    """Global configuration.

    :param dark_mode: Whether or not dark mode is enabled.
    :param plotly_template: Plotly template to use for charts in the UI.
    :param polars_engine: Polars computation backend.
    """

    @enum.unique
    class PlotlyTemplate(enum.StrEnum):
        ggplot2 = enum.auto()
        gridon = enum.auto()
        plotly = enum.auto()
        plotly_dark = enum.auto()
        plotly_white = enum.auto()
        presentation = enum.auto()
        seaborn = enum.auto()
        simple_white = enum.auto()
        xgridoff = enum.auto()
        ygridoff = enum.auto()


    @enum.unique
    class PolarsEngine(enum.StrEnum):
        gpu = enum.auto()
        streaming = enum.auto()

    dark_mode: bool = False
    plotly_template: PlotlyTemplate = PlotlyTemplate.plotly
    polars_engine: PolarsEngine = PolarsEngine.streaming


# global configuration
SETTINGS = Settings()