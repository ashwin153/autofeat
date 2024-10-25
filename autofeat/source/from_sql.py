import functools
from collections.abc import Iterator

import connectorx
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
    def load_data(
        with_columns: list[str] | None,
        predicate: polars.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[polars.DataFrame]:
        query = f"SELECT {', '.join(with_columns) if with_columns else '*'} FROM {table_name}"

        if n_rows is not None:
            query += f" LIMIT {n_rows}"

        table = connectorx.read_sql(uri, query, return_type="arrow2")
        assert isinstance(table, pyarrow.Table)

        for batch in table.to_batches(batch_size):
            df = polars.from_arrow(batch, schema)
            assert isinstance(df, polars.DataFrame)

            # TODO: push predicates down to the where clause
            if predicate is not None:
                df = df.filter(predicate)

            yield df

    return polars.io.plugins.register_io_source(
        callable=load_data,
        schema=schema,
    )


@functools.cache
def _load_schemas(
    uri: str,
) -> dict[str, polars.Schema]:
    engine = sqlalchemy.create_engine(uri)
    metadata = sqlalchemy.MetaData()
    metadata.reflect(engine)
    engine.dispose()

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
