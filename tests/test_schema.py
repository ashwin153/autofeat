from autofeat import Attribute, Schema


def test_select() -> None:
    schema = Schema({
        "x": {Attribute.numeric, Attribute.primary_key},
        "y": {Attribute.numeric},
        "z": {Attribute.textual},
    })

    assert {"x", "y"} == set(schema.select(include={Attribute.numeric}))
    assert {"y"} == set(schema.select(include={Attribute.numeric}, exclude={Attribute.primary_key}))
