from autofeat.transform import Aggregate, Cast, Encode, Identity, Transform


def transform_editor(

) -> Transform:
    """Configure the transform used by feature selection.

    :return: Transform.
    """
    # TODO: make this configurable
    return (
        Cast()
        .then(Encode())
        .then(Identity(), Aggregate())
    )
