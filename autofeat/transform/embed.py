import dataclasses
from collections.abc import Iterable
from typing import Literal

import polars
import polars_candle  # noqa: F401

from autofeat.attribute import Attribute
from autofeat.convert import into_exprs, into_named_exprs
from autofeat.table import Column, Table
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
        for table in tables:
            if embeddings := list(self._embeddings(table)):
                extra_columns = [
                    column
                    for column in table.columns
                    if all(column.name != embedded_column.name for embedded_column, _ in embeddings)
                ]

                columns = [
                    *extra_columns,
                    *[column for column, _ in embeddings],
                ]

                yield Table(
                    data=table.data.select(
                        *into_exprs(extra_columns),
                        **into_named_exprs(embeddings),
                    ),
                    name=f"embed({table.name})",
                    columns=columns,
                )

    def _embeddings(
        self,
        table: Table,
    ) -> Iterable[tuple[Column, polars.Expr]]:
        for column in table.columns:
            if (
                Attribute.textual in column.attributes
                and Attribute.categorical not in column.attributes
            ):
                result = Column(
                    name=column.name,
                    attributes={Attribute.embedding},
                    derived_from=[(column, table)],
                )

                yield result, column.expr.candle.embed_text(  # type: ignore[attr-defined]
                    device=self.device,
                    model_repo=self.model,
                    normalize=self.normalize,
                    pooling=self.pooling,
                )
