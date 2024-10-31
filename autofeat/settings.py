import dataclasses
from typing import Literal

import cuda.cudart

# use the cuda runtime to check if cuda is available
_CUDA_IS_AVAILABLE = cuda.cudart.cudaGetDeviceCount()[1] > 0


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
    :param low_memory: Whether or not to conserve memory at the expense of performance.
    :param plotly_template: Plotly template to use for charts in the UI.
    :param polars_engine: Polars computation backend.
    """

    dark_mode: bool = False
    display_mode: DisplayMode = "minimal"
    low_memory: bool = False
    plotly_template: PlotlyTemplate = "plotly"
    polars_engine: PolarsEngine = "gpu" if _CUDA_IS_AVAILABLE else "streaming"

    def __post_init__(
        self,
    ) -> None:
        assert (
            self.polars_engine != "gpu" or _CUDA_IS_AVAILABLE
        ), "cuda must be available to use the gpu"


# global configuration
SETTINGS = Settings()
