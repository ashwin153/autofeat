from autofeat.attribute import Attribute
from autofeat.schema import Schema


def test_select() -> None:
    schema = Schema({
        "x": {Attribute.numeric, Attribute.primary_key},
        "y": {Attribute.numeric},
        "z": {Attribute.textual},
    })

    assert {"x", "y"} == schema.select(
        include={Attribute.numeric},
    )

    assert {"y"} == schema.select(
        include={Attribute.numeric},
        exclude={Attribute.primary_key},
    )
