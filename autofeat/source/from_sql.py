import functools
from collections.abc import Iterable, Iterator
from typing import cast

import polars.io.plugins
import pyarrow
import sqlalchemy

from autofeat.convert import into_columns
from autofeat.dataset import Dataset
from autofeat.table import Table


def from_sql(
    uri: str,
) -> Dataset:
    """Load from a SQL database.

    :param uri: Database connection URI.
    :return: SQL dataset.
    """
    tables = []

    schemas = _load_schemas(uri)

    for table_name, schema in schemas.items():
        data = _scan_data(uri, table_name, schema)

        table = Table(
            data=data,
            columns=into_columns(data),
            name=table_name,
        )

        tables.append(table)

    return Dataset(tables)


def _scan_data(
    uri: str,
    table_name: str,
    schema: polars.Schema,
) -> polars.LazyFrame:
    def source(
        with_columns: list[str] | None,
        predicate: polars.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[polars.DataFrame]:
        query = f"SELECT {', '.join(with_columns) if with_columns else '*'} FROM {table_name}"

        if n_rows is not None:
            query += f" LIMIT {n_rows}"

        for df in _load_data(uri, query, batch_size):
            # TODO: push predicates into the query
            if predicate is not None:
                df = df.filter(predicate)

            yield df

    return polars.io.plugins.register_io_source(
        callable=source,
        schema=schema,
    )


def _load_data(
    uri: str,
    query: str,
    batch_size: int | None,
) -> Iterable[polars.DataFrame]:
    try:
        # TODO: connectorx supports a subset of backends (e.g., not snowflake)
        import connectorx

        table = connectorx.read_sql(uri, query, return_type="arrow2")
        assert isinstance(table, pyarrow.Table)

        for batch in table.to_batches(batch_size):
            yield cast(polars.DataFrame, polars.from_arrow(batch))
    except ImportError:
        engine = sqlalchemy.create_engine(uri)

        with engine.connect() as connection:
            return polars.read_database(
                query,
                connection=connection,
                iter_batches=True,
                batch_size=batch_size,
            )


@functools.cache
def _load_schemas(
    uri: str,
) -> dict[str, polars.Schema]:
    engine = sqlalchemy.create_engine(uri)
    metadata = sqlalchemy.MetaData()
    metadata.reflect(engine)

    return {
        table.name: polars.Schema({
            column.name: _into_data_type(column.type)
            for column in table.columns.values()
        })
        for table in metadata.tables.values()
    }


def _into_data_type(
    column_type: sqlalchemy.types.TypeEngine,
) -> polars.DataType:
    if isinstance(column_type, sqlalchemy.types.String):
        return polars.String()
    elif isinstance(column_type, sqlalchemy.types.Boolean):
        return polars.Boolean()
    elif isinstance(column_type, sqlalchemy.types.Integer):
        return polars.Int64()
    elif isinstance(column_type, sqlalchemy.types.Numeric):
        return polars.Float64()
    else:
        raise NotImplementedError(f"{column_type} is not supported")
