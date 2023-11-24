import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.dbconnect import DbConnect

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


# Inititalize the Data Ingestion Configuration...


@dataclass
class DataIngestionconfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")

# create a class for Data Ingestion


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion methods starts..")
        try:
            df1 = pd.read_csv(os.path.join(
                "notebooks", "income_cleaned.csv"))  # data coming from csv pregenerated...

            obj = DbConnect()  # casandra db connection...
            df2 = pd.DataFrame(obj.load_data())  # data coming from database

            df = df2
            logging.info(df1)
            logging.info("df22222222222222222222222222222222")
            logging.info(df2)

            logging.info("DataSet read as Pandas Dataframe")

            os.makedirs(os.path.dirname(
                self.ingestion_config.raw_data_path), exist_ok=True)

            # raw.csv will be created....
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info(("Train test split....."))

            train_set, test_set = train_test_split(
                df, test_size=0.30, random_state=42)

            # train.csv and test.csv will be create in artifacts
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,
                            index=False, header=True)

            logging.info("Data Ingestion is completed....")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception occured at Data Ingestion stage...")
            raise CustomException(e, sys)


# run data ingestion

# if __name__ == '__main__':
    # etc
# if __name__ == "__main__":
#     obj = DataIngestion()
#     train_data_path, test_data_path = obj.initiate_data_ingestion()
    # data_transformation1 = DataTransformation()
    # train_arr, test_arr, _ = data_transformation1.initiate_data_transformation(
    #     train_data_path, test_data_path)
