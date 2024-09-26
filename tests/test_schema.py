from autofeat.attribute import Attribute
from autofeat.schema import Schema

SCHEMA = Schema({
    "x": {Attribute.numeric, Attribute.primary_key},
    "y": {Attribute.numeric},
    "z": {Attribute.textual},
})


def test_iter() -> None:
    assert ["x", "y", "z"] == list(iter(SCHEMA))


def test_select() -> None:
    assert {"x", "y"} == SCHEMA.select(
        include={Attribute.numeric},
    )

    assert {"y"} == SCHEMA.select(
        include={Attribute.numeric},
        exclude={Attribute.primary_key},
    )
