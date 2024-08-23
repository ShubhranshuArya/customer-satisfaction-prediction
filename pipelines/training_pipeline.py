from zenml import pipeline

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_trainer import model_trainer
from steps.evaluate_model import evaluate_model


@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    df = ingest_df(data_path)
    clean_df(df)
    model_trainer(df)
    evaluate_model(df)
