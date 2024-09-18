from collections.abc import Iterable

import polars

from autofeat.action.extract_filters import IntoFilters, extract_filters
from autofeat.action.extract_tables import IntoTables, extract_tables
from autofeat.transform.identity import Identity


def extract_features(
    *,
    tables: IntoTables | Iterable[IntoTables],
    filters: IntoFilters | Iterable[IntoFilters],
) -> polars.LazyFrame:
    """Extract features from the ``tables`` for each of the ``filters``.

    Features are the boolean or numeric columns in ``tables`` that contain a single row. A simple
    way to guarantee this is to aggregate the ``tables`` by the same columns that they are filtered
    by.

    :param tables: Tables to extract features from.
    :param filters: Filters to apply before transforming the data.
    :return: Extracted features.
    """
    feature_vectors = []
    tables = extract_tables(tables)
    transforms = extract_filters(filters) or (Identity(),)

    for transform in transforms:
        feature_values = []

        for table in transform.apply(tables):
            feature_selector = (
                (polars.selectors.boolean() | polars.selectors.numeric())
                .name.suffix(f" from {table.name}")
            )

            feature_values.append(
                table.data
                .filter(polars.len() == 1)
                .select(feature_selector),
            )

        feature_vectors.append(polars.concat(feature_values, how="horizontal"))

    return polars.concat(feature_vectors, how="diagonal")
