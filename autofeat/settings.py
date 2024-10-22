import dataclasses


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Settings:
    """Application settings.

    :param chart_theme: Plotly template to use in charts.
    :param dark_mode: Whether or not dark mode is enabled.
    """

    chart_theme: str
    dark_mode: bool
