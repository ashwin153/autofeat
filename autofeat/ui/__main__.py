import streamlit
import streamlit.runtime.caching.hashing

from autofeat.ui.dataset_explorer import dataset_explorer
from autofeat.ui.dataset_loader import dataset_loader
from autofeat.ui.feature_loader import feature_loader

streamlit.set_page_config(
    page_title="autofeat",
    layout="wide",
)

if dataset := dataset_loader():
    dataset_explorer(dataset)
    feature_loader(dataset)
