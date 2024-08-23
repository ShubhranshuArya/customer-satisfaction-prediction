import logging
import pandas as pd
from zenml import step


@step
def model_trainer(df: pd.DataFrame):
    pass
