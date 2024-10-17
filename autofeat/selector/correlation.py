from typing import Literal

import attrs
import numpy
import pandas

from autofeat.selector.base import Selector


@attrs.define(frozen=True, kw_only=True)
class Correlation(Selector):
    """Select features that are at most ``max`` correlated with any other selected feature.

    :param max: Maximum correlation between selected features.
    :param method: Correlation method.
    """

    max: float = 0.5
    method: Literal["pearson", "kendall", "spearman"] = "pearson"

    def select(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
    ) -> list[bool]:
        correlation = numpy.max(
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

        selection = numpy.argwhere(correlation < self.max)

        return [
            i in selection
            for i in range(X.shape[1])
        ]
