import pytest

from autofeat.transform import AllOf, AnyOf, Cast, Combine, Encode, Identity, Transform


@pytest.mark.parametrize(
    "actual,expected",
    [
        (
            Identity().then(Identity()),
            Identity(),
        ),
        (
            Identity().then(Cast()),
            Cast(),
        ),
        (
            Cast().then(Identity()),
            Cast(),
        ),
        (
            Cast().then(Combine()).then(Encode()),
            AllOf([Cast(), Combine(), Encode()]),
        ),
        (
            Cast().then(Combine(), Identity()),
            AnyOf([AllOf([Cast(), Combine()]), Cast()]),
        ),
    ],
)
def test_then(
    actual: Transform,
    expected: Transform,
) -> None:
    assert actual == expected
