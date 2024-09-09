import abc
import dataclasses
import functools
from typing import Iterable

import polars


@dataclasses.dataclass(frozen=True, kw_only=True)
class Table:
    """A lazily-loaded, tabular data.
    
    :param data: Table contents.
    """

    data: polars.LazyFrame

    @functools.cached_property
    def schema(
        self,
    ) -> polars.Schema:
        """Describe the columns in this table.
        
        :return: Table columns.
        """
        return self.data.collect_schema()


class Dataset(abc.ABC):
    """A collection of tables."""

    @abc.abstractmethod
    def list_tables(
        self,
    ) -> Iterable[Table]:
        """List all tables in this dataset.
        
        :return: All tables.
        """
