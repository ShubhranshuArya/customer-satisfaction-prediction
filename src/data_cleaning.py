import logging
from abc import ABC, abstractmethod
from typing_extensions import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract class to define the strategy for Data Cleaning
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        pass


class DataPreProcessStrategy(DataStrategy):
    """
    Pre-processing steps on the dataset.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-Processing dataset using the abstract class
        """
        try:
            # Drop un-necessary features
            data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                    "customer_zip_code_prefix",
                    "order_item_id",
                ],
                axis=1,
            )
            # Handle null values
            data["product_weight_g"].fillna(
                data["product_weight_g"].median(),
                inplace=True,
            )
            data["product_length_cm"].fillna(
                data["product_length_cm"].median(),
                inplace=True,
            )
            data["product_height_cm"].fillna(
                data["product_height_cm"].median(),
                inplace=True,
            )
            data["product_width_cm"].fillna(
                data["product_width_cm"].median(),
                inplace=True,
            )
            ## write "No review" in review_comment_message column
            data["review_comment_message"].fillna(
                "No review",
                inplace=True,
            )
            # Include only numerical features
            data = data.select_dtypes(include=[np.number])

            return data

        except Exception as e:
            logging.error(e)
            raise e


class DataSplitStrategy(DataStrategy):
    """
    Train test splitting of the dataset
    """

    def handle_data(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the dataset based on certain criteria.

        Parameters:
        data (pd.DataFrame): The input dataframe containing the dataset to be split.

        Returns:
        Union[pd.DataFrame, pd.Series]: The split data as either a DataFrame or Series.

        """
        try:
            # Separate dependent and Independent features
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            # Split the features
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(e)
            raise e


class DataCleaning:
    """
    Class which implements data pre processing and data splitting
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        self.data = data
        self.strategy = strategy

    def handle_data(
        self,
    ) -> Union[
        pd.DataFrame, pd.Series, tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    ]:
        return self.strategy.handle_data(self.data)
