import collections
import dataclasses
from collections.abc import Iterable

import polars
import sklearn.feature_selection

from autofeat.convert import IntoDataFrame, IntoSeries, into_series
from autofeat.solver import Model
from autofeat.table import Table
from autofeat.transform.base import Transform
from autofeat.transform.extract import Extract
from autofeat.transform.keep import Keep

SEPARATOR = "::@::"


@dataclasses.dataclass(frozen=True, kw_only=True)
class Select(Transform):
    """Select the most predictive features.

    :param known: Known variables.
    :param limit: Maximum number of features to select.
    :param model: Model used to measure feature importance.
    :param target: Target variable.
    """

    known: IntoDataFrame
    limit: int
    model: Model
    target: IntoSeries

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        # load training data
        tables = list(tables)

        features = polars.concat(
            polars.collect_all(
                [
                    table.data.select(polars.all().name.prefix(f"{table.name}{SEPARATOR}"))
                    for table in Extract(known=self.known).apply(tables)
                ],
            ),
            how="horizontal",
        )

        target = into_series(self.target)

        # fit a model
        selector = sklearn.feature_selection.SelectFromModel(
            self.model,
            max_features=self.limit,
        )

        selector.fit(
            features.to_numpy(),
            target.to_numpy(),
        )

        # select the most important features
        selected: dict[str, set[str]] = collections.defaultdict(set)

        for i, selected in enumerate(selector.get_support()):
            if selected:
                table_name, column = features.columns[i].split(SEPARATOR, 1)
                selected[table_name].add(column)

        # keep only the selected features
        yield from Keep(columns=selected).apply(tables)
