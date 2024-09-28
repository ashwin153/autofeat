import pathlib

import streamlit


@streamlit.cache_data
def readme() -> str:
    readme = pathlib.Path(__file__).parent.parent / "README.md"
    return readme.read_text()


streamlit.markdown(readme())
