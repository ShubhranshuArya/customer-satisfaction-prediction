import logging
import pandas as pd
from zenml import step
from typing_extensions import Annotated
from src.data_cleaning import DataCleaning, DataPreProcessStrategy, DataSplitStrategy


@step
def clean_df(df: pd.DataFrame) -> tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    try:
        data_pre_process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, strategy=data_pre_process_strategy)
        pre_processed_data = data_cleaning.handle_data()

        data_split_strategy = DataSplitStrategy()
        data_cleaning = DataCleaning(pre_processed_data, strategy=data_split_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data Cleaning Completed")

    except Exception as e:
        logging.error(e)
        raise e
