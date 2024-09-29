import streamlit
import streamlit.runtime.caching.hashing

from autofeat.ui.dataset_explorer import dataset_explorer
from autofeat.ui.dataset_loader import dataset_loader
from autofeat.ui.feature_loader import feature_loader
from autofeat.ui.schema_editor import schema_editor

streamlit.set_page_config(
    page_title="autofeat",
    layout="wide",
)

if original_dataset := dataset_loader():
    dataset_explorer(original_dataset)
    edited_dataset = schema_editor(original_dataset)
    feature_loader(edited_dataset)
