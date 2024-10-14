import pathlib

import streamlit.testing.v1


def test_app() -> None:
    main = pathlib.Path(__file__).parent.parent / "autofeat" / "ui" / "__main__.py"

    app = streamlit.testing.v1.AppTest.from_file(str(main))

    app.run()

    assert not app.exception
