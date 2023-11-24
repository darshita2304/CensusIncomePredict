import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectPercentile, chi2
import sklearn.preprocessing

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

from src.components.data_ingestion import DataIngestion


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation initiated...")

            # # Define which columns should be ordinal-encoded and which should be scaled
            # # column list for ordinal ecoding
            # # cat_ordinal_cols = ['workclass']
            # # column list for label encoding
            # cat_label_cols = ['workclass', 'occupation', 'relationship']

            # # column for scaler  encoding
            # numerical_cols = ['age', 'fnlwgt',
            #                   'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

            # # Define the custom ranking for each ordinal variable
            # # workclass_categories = [
            # # 'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', 'Jobless', 'Self-emp-inc', 'Without-pay', 'Never-worked']

            # logging.info("Pipeline Initiated....")
            # # Numerical Pipeline
            # num_pipeline = Pipeline(
            #     steps=[
            #         ('imputer', SimpleImputer(strategy='median'), numerical_cols)
            #     ]
            # )
            # # cat_ord_pipeline = Pipeline(steps=[
            # #     ('ordinalimputer', OrdinalEncoder(
            # #         categories=[workclass_categories]), cat_ordinal_cols)
            # # ])

            # cat_pipeline = Pipeline(
            #     steps=[('lableencoder', OneHotEncoder(
            #         handle_unknown="ignore"), cat_label_cols)]

            # )
            # scale_pipeline = Pipeline(steps=[('scale', StandardScaler())])

            # # preprocessor = Pipeline(steps=[("categorical_ord", cat_ord_pipeline, cat_ordinal_cols),
            # #                                ("categorical_lbl",
            # #                                 cat_lbl_pipeline, cat_label_cols),
            # #                                ("numerical", num_pipeline, numerical_cols)])

            # preprocessor = ColumnTransformer(transformers=[("num_pipeline", num_pipeline),
            #                                                ("cat_pipeline",
            #                                                 cat_pipeline),
            #                                                ("scale_pipeline", scale_pipeline)])

            numeric_features = ['age', 'fnlwgt', 'education_num',
                                'capital_gain', 'capital_loss', 'hours_per_week']
            numeric_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="median")),
                       ("scaler", StandardScaler())]
            )

            categorical_features = ['workclass', 'occupation', 'relationship']
            categorical_transformer = Pipeline(
                steps=[
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", SelectPercentile(chi2, percentile=50)),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features),
                    ("cat", categorical_transformer, categorical_features),
                ]
            )

            logging.info("Pipeline Completed..")

            return preprocessor

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(
                f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(
                f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessor = self.get_data_transformation_object()  # this fn is within class

            target_column_name = 'fiftyplus'

            # dropping two very less corelated columns along with target..
            drop_columns = [target_column_name,
                            'education', 'sex', 'fnlwgt', 'race', 'marital_status', 'native_country']

            target_feature_train_df = train_df[target_column_name]
            train_df.drop(columns=drop_columns, axis=1)
            input_feature_train_df = train_df

            target_feature_test_df = test_df[target_column_name]
            test_df.drop(columns=drop_columns, axis=1)
            input_feature_test_df = test_df

            # Trnasformating using preprocessor obj
            # print(input_feature_train_df)

            input_feature_train_arr = preprocessor.fit_transform(
                input_feature_train_df, target_feature_train_df)
            input_feature_test_arr = preprocessor.transform(
                input_feature_test_df)

            logging.info(
                "Applying preprocessing object on training and testing datasets.")

            # converting data in array to load array very quickly...
            # numpy array are superfast for ML
            train_arr = np.c_[input_feature_train_arr,
                              np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,
                             np.array(target_feature_test_df)]

            # saving pickle file from preprocessor object..
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            # return (train_arr, test_arr)

        except Exception as e:
            logging.info("Exception occured in initiate_data_transformation")
            raise CustomException(e, sys)


# if __name__ == "__main__":
#     obj = DataIngestion()
#     train_data_path, test_data_path = obj.initiate_data_ingestion()
#     data_transformation1 = DataTransformation()
#     train_arr, test_arr, _ = data_transformation1.initiate_data_transformation(
#         train_data_path, test_data_path)
