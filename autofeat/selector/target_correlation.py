import dataclasses
from typing import Literal

import numpy
import pandas

from autofeat.selector.base import Selector


@dataclasses.dataclass(kw_only=True)
class TargetCorrelation(Selector):
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
        target_correlation = numpy.abs(
            pandas.DataFrame(X)
            .corrwith(pandas.Series(y), axis=1, method=self.method)
            .to_numpy(),
        )

        selection = numpy.argwhere(target_correlation < self.threshold)

        return [
            i in selection
            for i in range(X.shape[1])
        ]
