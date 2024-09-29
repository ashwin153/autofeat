import dataclasses
from collections.abc import Iterable

from autofeat.attribute import Attribute
from autofeat.convert import IntoDataFrame, into_data_frame
from autofeat.schema import Schema
from autofeat.table import Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Extract(Transform):
    """Extract features that are relevant to the ``known`` data.

    :param known: Data that is already known.
    """

    known: IntoDataFrame

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        known = into_data_frame(self.known)

        for table in tables:
            primary_key = list(table.schema.select(include={Attribute.primary_key}))

            if primary_key and all(column in known.columns for column in primary_key):
                schema = Schema({
                    **table.schema.select(
                        include={Attribute.boolean},
                        exclude={Attribute.primary_key},
                    ),
                    **table.schema.select(
                        include={Attribute.numeric},
                        exclude={Attribute.primary_key},
                    ),
                })

                data = (
                    known
                    .lazy()
                    .join(table.data, on=primary_key, how="left")
                    .select(schema.keys())
                )

                yield Table(
                    data=data,
                    name=table.name,
                    schema=schema,
                )
