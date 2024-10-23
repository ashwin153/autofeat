import functools
from collections.abc import Iterator
from typing import TYPE_CHECKING

import connectorx
import polars.io.plugins
import sqlalchemy

from autofeat.convert import into_columns
from autofeat.dataset import Dataset
from autofeat.table import Table

if TYPE_CHECKING:
    import pyarrow


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
    def data(
        with_columns: list[str] | None,
        predicate: polars.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[polars.DataFrame]:
        query = f"SELECT {', '.join(with_columns) if with_columns else '*'} FROM {table_name}"

        if n_rows is not None:
            query += f" LIMIT {n_rows}"

        table: pyarrow.Table = connectorx.read_sql(
            conn=uri,
            query=query,
            return_type="arrow2",
        )

        for batch in table.to_batches(batch_size):
            df = polars.from_arrow(batch)

            # TODO: push predicates down to the where clause
            if predicate is not None:
                df = df.filter(predicate)

            yield df

    return polars.io.plugins.register_io_source(
        callable=data,
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
    # TODO: convert between sqlalchemy and polars types
    ...
