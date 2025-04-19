import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    selected_features_file_path = os.path.join('artifacts', "selected_features.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["BMI","Dia_BP","OGTT"]
            categorical_columns = ["PCOS","Prediabetes"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def select_features(self, X, y, method="SelectKBest", k=5, feature_names=None):
        try:
            if method == "RFE":
                selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=k)
                X_selected = selector.fit_transform(X, y)
                selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
                return X_selected, selected_features
            elif method == "SelectKBest":
                selector = SelectKBest(score_func=f_classif, k=k)
                X_selected = selector.fit_transform(X, y)
                selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
                return X_selected, selected_features
            elif method == "RFC":
                model = RandomForestClassifier()
                model.fit(X, y)
                importances = model.feature_importances_
                indices = np.argsort(importances)[-k:]
                X_selected = X[:, indices]
                selected_features = [feature_names[i] for i in indices]
                return X_selected, selected_features
            elif method == "PFI":
                model = GradientBoostingClassifier()
                model.fit(X, y)
                result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=2)
                indices = result.importances_mean.argsort()[-k:]
                X_selected = X[:, indices]
                selected_features = [feature_names[i] for i in indices]
                return X_selected, selected_features
            elif method == "ANOVA":
                selector = SelectKBest(score_func=f_classif, k=k)  # Initialize SelectKBest to select top k features
                X_selected = selector.fit_transform(X, y)  # Fit the model and transform the dataset
                indices = selector.get_support(indices=True)
                selected_features = [feature_names[i] for i in indices]
                return X_selected, selected_features
            elif method == "MutualInfo":
                selector = SelectKBest(score_func=mutual_info_classif, k=k)  # Initialize SelectKBest to select top k features
                X_selected = selector.fit_transform(X, y)  # Fit the model and transform the dataset
                indices = selector.get_support(indices=True)
                selected_features = [feature_names[i] for i in indices]
                return X_selected, selected_features
            elif method == "TreeBased":
                models = [
                    RandomForestClassifier(),
                    GradientBoostingClassifier(),
                    ExtraTreesClassifier(),
                    DecisionTreeClassifier()
                ]
                importances = np.zeros(X.shape[1])
                for model in models:
                    model.fit(X, y)  # Fit the model on the entire dataset
                    importances += model.feature_importances_  # Sum the importances
                importances /= len(models)
                indices = np.argsort(importances)[-k:][::-1]
                X_selected = X[:, indices]
                selected_features = [feature_names[i] for i in indices]
                return X_selected, selected_features
            else:
                raise ValueError("Unsupported feature selection method")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path, feature_selection_method="SelectKBest", k=5):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Have Read train & test Data")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Result"
            numerical_columns = ["BMI","Dia_BP","OGTT"]
            all_columns = numerical_columns + ["PCOS","Prediabetes"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info(f"Applying feature selection method: {feature_selection_method}")
            input_feature_train_arr, selected_features = self.select_features(input_feature_train_arr, target_feature_train_df, method=feature_selection_method, k=k, feature_names=all_columns)
            input_feature_test_arr = input_feature_test_arr[:, [all_columns.index(f) for f in selected_features]]

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved Preprocessing Object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                selected_features
            )

        except Exception as e:
            raise CustomException(e, sys)
