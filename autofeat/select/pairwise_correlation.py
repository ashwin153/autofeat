import dataclasses
from typing import Literal

import numpy
import pandas

from autofeat.select.base import Selector


@dataclasses.dataclass(kw_only=True)
class PairwiseCorrelation(Selector):
    """Select features that are at most ``max`` correlated with any other selected feature.

    :param method: Correlation method.
    :param threshold: Maximum correlation between selected features.
    """

    method: Literal["pearson", "kendall", "spearman"] = "pearson"
    threshold: float = 0.5

    def select(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
    ) -> list[bool]:
        pairwise_correlation = numpy.max(
            numpy.triu(
                numpy.abs(
                    pandas.DataFrame(X)
                    .corr(self.method)
                    .to_numpy(),
                ),
                k=1,
            ),
            axis=1,
        )

        selection = numpy.argwhere(pairwise_correlation < self.threshold)

        return [
            i in selection
            for i in range(X.shape[1])
        ]
