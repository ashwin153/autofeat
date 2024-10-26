import dataclasses
from collections.abc import Iterable
from typing import Literal

from autofeat.table import Table
from autofeat.transform import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Embed(Transform):
    """Embed text columns as vectors.

    .. _MTEB Leaderboard:
        https://huggingface.co/spaces/mteb/leaderboard

    :param device: Device to use.
    :param model: Fully qualified name of the text embedding model.
    :param normalize: Whether or not to L2 normalize the embeddings.
    :param pooling: Pooling strategy to use.
    """

    # TODO: add this to settings instead of hard-coding everywhere
    device: Literal["cpu", "gpu"] = "cpu"
    model: str = "dunzhang/stella_en_400M_v5"
    normalize: bool = False
    pooling: Literal["max", "sum", "mean"] = "mean"

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        ...
