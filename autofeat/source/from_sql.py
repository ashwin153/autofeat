import functools
from collections.abc import Iterable, Iterator
from typing import cast

import loguru
import polars.io.plugins
import pyarrow
import sqlalchemy

from autofeat.attribute import Attribute
from autofeat.dataset import Dataset
from autofeat.table import Column, Table

try:
    import connectorx

    _CONNECTORX_IS_INSTALLED = True
except ImportError:
    _CONNECTORX_IS_INSTALLED = False


def from_sql(
    uri: str,
    *,
    schema: str | None = None,
) -> Dataset:
    """Load from a SQL database.

    :param uri: Database connection URI.
    :param schema: Database schema to connect to.
    :return: SQL dataset.
    """
    return Dataset(list(_load_tables(uri, schema)))


@functools.cache
def _load_tables(
    uri: str,
    schema: str | None,
) -> Iterable[Table]:
    loguru.logger.info("loading metadata")
    engine = sqlalchemy.create_engine(uri)
    metadata = sqlalchemy.MetaData()
    metadata.reflect(engine, schema=schema)
    loguru.logger.info("loaded metadata")

    with engine.connect() as connection:
        for table in metadata.tables.values():
            table_name = (
                f"{schema}.{table.name}"
                if schema
                else table.name
            )

            try:
                len = connection.execute(f"""
                    SELECT COUNT(*)
                    FROM {table_name}
                """).first()[0]

                columns = [
                    Column(
                        name=column.name,
                        attributes=Attribute.infer(
                            data_type=_into_data_type(column.type),
                            len=len,
                            n_unique=connection.execute(f"""
                                SELECT COUNT(DISTINCT {column.name})
                                FROM {table_name}
                            """).first()[0],
                            null_count=connection.execute(f"""
                                SELECT COUNT(*)
                                FROM {table_name}
                                WHERE {column.name} IS NULL
                            """).first()[0],
                        ),
                    )
                    for column in table.columns.values()
                ]

                data = _scan_data(uri, table_name, table)

                yield Table(
                    data=data,
                    columns=columns,
                    name=table_name,
                )
            except Exception:
                loguru.logger.exception(f"failed to load table {table_name}")


def _scan_data(
    uri: str,
    name: str,
    table: sqlalchemy.Table,
) -> polars.LazyFrame:
    def source(
        with_columns: list[str] | None,
        predicate: polars.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[polars.DataFrame]:
        query = f"SELECT {', '.join(with_columns) if with_columns else '*'} FROM {name}"

        if n_rows is not None:
            query += f" LIMIT {n_rows}"

        for df in _load_data(uri, query, batch_size):
            # TODO: push predicates into the query
            if predicate is not None:
                df = df.filter(predicate)

            yield df

    return polars.io.plugins.register_io_source(
        callable=source,
        schema=_into_schema(table),
    )


# https://github.com/sfu-db/connector-x/tree/main/connectorx/src/sources
_CONNECTORX_SOURCES = (
    "bigquery://",
    "mssql://",
    "mysql://",
    "oracle://",
    "postgresql://",
    "redshift://",
    "sqlite://",
    "trino://",
)


def _load_data(
    uri: str,
    query: str,
    batch_size: int | None,
) -> Iterable[polars.DataFrame]:
    if _CONNECTORX_IS_INSTALLED and uri.startswith(_CONNECTORX_SOURCES):
        table = connectorx.read_sql(uri, query, return_type="arrow2")
        assert isinstance(table, pyarrow.Table)

        for batch in table.to_batches(batch_size):
            yield cast(polars.DataFrame, polars.from_arrow(batch))
    else:
        engine = sqlalchemy.create_engine(uri)

        with engine.connect() as connection:
            yield from polars.read_database(
                query,
                connection=connection,
                iter_batches=True,
                batch_size=batch_size,
            )


def _into_schema(
    table: sqlalchemy.Table,
) -> polars.Schema:
    return polars.Schema({
        column.name: _into_data_type(column.type)
        for column in table.columns.values()
    })


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
    elif isinstance(column_type, sqlalchemy.types.Date):
        return polars.Date()
    elif isinstance(column_type, sqlalchemy.types.DateTime):
        return polars.Datetime()
    else:
        raise NotImplementedError(f"{column_type} is not supported")
