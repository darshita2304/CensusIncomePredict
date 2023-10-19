import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            logging.info("scaling.......")
            data_scaled = preprocessor.transform(features)
            logging.info(data_scaled)

            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 age: float,
                 workclass: str,
                 education: int,
                 fnlwgt: int,
                 occupation: str,
                 relationship: str,
                 capital_gain: float,
                 capital_loss: float,
                 hours_per_week: float
                 ):

        self.age = age
        self.workclass = workclass
        self.education = education
        self.occupation = occupation
        self.fnlwgt = fnlwgt
        self.relationship = relationship
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'age': [self.age],
                'workclass': [self.workclass],
                'education-num': [self.education],
                'fnlwgt': [self.fnlwgt],
                'occupation': [self.occupation],
                'relationship': [self.relationship],
                'capital-gain': [self.capital_gain],
                'capital-loss': [self.capital_loss],
                'hours-per-week': [self.hours_per_week]

            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df

        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e, sys)
