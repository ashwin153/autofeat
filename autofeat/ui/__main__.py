import streamlit

from autofeat.ui.combine_features import combine_features
from autofeat.ui.edit_dataset import edit_dataset
from autofeat.ui.edit_settings import edit_settings
from autofeat.ui.evaluate_model import evaluate_model
from autofeat.ui.explore_dataset import explore_dataset
from autofeat.ui.explore_features import explore_features
from autofeat.ui.explore_predictions import explore_predictions
from autofeat.ui.load_dataset import load_dataset
from autofeat.ui.train_model import train_model

streamlit.set_page_config(
    initial_sidebar_state="collapsed",
    layout="wide",
    page_title="autofeat",
)

settings = edit_settings()

streamlit.header("Setup Dataset")
if dataset := load_dataset():
    dataset = edit_dataset(dataset)
    explore_dataset(dataset, settings)

    streamlit.header("Train Model")
    if model := train_model(dataset):
        streamlit.header("Explore Model")
        evaluate_model(model)
        explore_features(model, settings)
        combine_features(model)
        explore_predictions(model)
