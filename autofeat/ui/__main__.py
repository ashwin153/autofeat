import streamlit

from autofeat.ui.edit_schema import edit_schema
from autofeat.ui.evaluate_model import evaluate_model
from autofeat.ui.explore_dataset import explore_dataset
from autofeat.ui.load_dataset import load_dataset
from autofeat.ui.train_model import train_model

streamlit.set_page_config(
    page_title="autofeat",
    layout="wide",
)


streamlit.header("Load Dataset")
if dataset := load_dataset():
    dataset = edit_schema(dataset)
    explore_dataset(dataset)

    streamlit.header("Train Model")
    if model := train_model(dataset):
        evaluate_model(model)
