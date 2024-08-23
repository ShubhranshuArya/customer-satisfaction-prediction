import logging

import pandas as pd
from zenml import step


class IngestData:
    """
    Ingesting Data from the data path
    """

    def __init__(self, data_path: str):
        """
        Args:
            data_path: path to the data
        """
        self.data_path = data_path

    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)


@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingesting data from the data path

    Args:
        data_path: path to the data
    Returns:
        pd.DataFrame: The ingested data
    """
    logging.info(f"Starting data ingestion from {data_path}")
    try:
        data_ingestor = IngestData(data_path=data_path)
        df = data_ingestor.get_data()
        logging.info("Data ingestion completed successfully")
        return df
    except Exception as e:
        logging.error(f"Error during data ingestion: {e}")
        raise e
