import streamlit

from autofeat.ui.explore_dataset import explore_dataset
from autofeat.ui.load_dataset import load_dataset
from autofeat.ui.train_model import train_model

streamlit.set_page_config(
    page_title="autofeat",
    layout="wide",
)

if dataset := load_dataset():
    explore_dataset(dataset)
    train_model(dataset)
