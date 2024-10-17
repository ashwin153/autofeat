from __future__ import annotations

import enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autofeat.predictor.base import Predictor


@enum.unique
class Problem(enum.Enum):
    """A kind of prediction problem."""

    classification = enum.auto()
    regression = enum.auto()

    def __str__(
        self,
    ) -> str:
        return self.name

    def baseline(
        self,
    ) -> Predictor:
        """Get the baseline method for this kind of problem.

        :return: Baseline method.
        """
        from autofeat.predictor.baseline import Baseline

        baseline = Baseline()
        return baseline.create(self)
