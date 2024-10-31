import dataclasses
import functools
from typing import Literal, cast

import cuda.cudart
import loguru

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


@functools.cache
def _cuda_is_available() -> bool:
    try:
        cuda_device_count = cast(int, cuda.cudart.cudaGetDeviceCount()[1])
    except Exception:
        cuda_device_count = 0

    loguru.logger.info(f"detected {cuda_device_count} cuda devices")

    return cuda_device_count > 0


@dataclasses.dataclass(kw_only=True)
class Settings:
    """Global configuration.

    :param dark_mode: Whether or not dark mode is enabled.
    :param display_mode: Density of display in the UI.
    :param low_memory: Whether or not to conserve memory at the expense of performance.
    :param plotly_template: Plotly template to use for charts in the UI.
    :param polars_engine: Polars computation backend.
    """

    dark_mode: bool = False
    display_mode: DisplayMode = "minimal"
    low_memory: bool = False
    plotly_template: PlotlyTemplate = "plotly"
    polars_engine: PolarsEngine = "gpu" if _cuda_is_available() else "streaming"

    def __post_init__(
        self,
    ) -> None:
        assert (
            self.polars_engine != "gpu" or _cuda_is_available()
        ), "cuda must be available to use the gpu"


# global configuration
SETTINGS = Settings()
