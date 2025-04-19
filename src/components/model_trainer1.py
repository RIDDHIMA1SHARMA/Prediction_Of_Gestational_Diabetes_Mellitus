import os
import sys
from dataclasses import dataclass
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    ensemble_model_file_path = os.path.join("artifacts", "ensemble_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split Training & Testing Input Data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            feature_names = ['BMI','PCOS','Dia_BP','OGTT','Prediabetes']
            
            # Print the shape and features
            print("Training data shape:", X_train.shape)
            print("Testing data shape:", X_test.shape)
            print("Training data shape:", y_train.shape)
            print("Testing data shape:", y_test.shape)
            print("Type of y_train:", type(y_train))

            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "XGBClassifier": XGBClassifier(random_state=42),
                "CatBoosting Classifier": CatBoostClassifier(random_state=42,verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
                "Extra Trees Classifier": ExtraTreesClassifier(random_state=42),
                "Voting Classifier": VotingClassifier(
                    estimators=[
                        ('rf', RandomForestClassifier(random_state=42)),
                        ('dt', DecisionTreeClassifier(random_state=42)),
                        ('gb', GradientBoostingClassifier(random_state=42)),
                        ('xgb', XGBClassifier(random_state=42)),
                        ('catboost', CatBoostClassifier(random_state=42,verbose=False)),
                        ('ada', AdaBoostClassifier(random_state=42)),
                        ('et', ExtraTreesClassifier(random_state=42))
                    ],
                    voting='soft'
                )
            }

            '''For Regression:-
            params={
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }
            '''

            params = {
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'criterion': ['gini', 'entropy'],
                    'max_features': ['sqrt', 'log2', None]
                },
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2', None]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'criterion': ['friedman_mse'],
                    'max_features': ['sqrt', 'log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "XGBClassifier": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [3, 4, 5, 6],
                    'gamma': [0, 0.1, 0.2, 0.3],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
                },
                "CatBoosting Classifier": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Classifier": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Extra Trees Classifier": {
                    'n_estimators': [8, 16, 32, 64, 100, 128, 256],
                },
                "Voting Classifier": {
                    'voting': ['hard', 'soft'],
                }
            }

            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            # For getting best model score from dict
            best_model_score = max(sorted(model_report.values()))
            # To get Best Model Name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.7:
                raise CustomException("No Best Model Found")

            logging.info(f"Best found model on both training & testing dataset: {best_model_name}")
            print(f"Best Single Model: {best_model_name}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=(best_model, feature_names)
            )

            # Save the ensemble model
            ensemble_model = models["Voting Classifier"]
            save_object(
                file_path=self.model_trainer_config.ensemble_model_file_path,
                obj=ensemble_model
            )

            # Evaluate the best model
            predicted_best_model = best_model.predict(X_test)
            acc_score = accuracy_score(y_test, predicted_best_model)
            precision = precision_score(y_test, predicted_best_model)
            recall = recall_score(y_test, predicted_best_model)
            f1 = f1_score(y_test, predicted_best_model)
            confusion_best_model = confusion_matrix(y_test, predicted_best_model)

            # Evaluate the ensemble model
            predicted_ensemble = ensemble_model.predict(X_test)
            ensemble_acc_score = accuracy_score(y_test, predicted_ensemble)
            ensemble_precision = precision_score(y_test, predicted_ensemble)
            ensemble_recall = recall_score(y_test, predicted_ensemble)
            ensemble_f1 = f1_score(y_test, predicted_ensemble)
            confusion_ensemble = confusion_matrix(y_test, predicted_ensemble)

            # Print hyperparameters used for the best model
            # Best model's hyperparameters
            print(f"Best Model Hyperparameters: {best_model.get_params()}")
            
            # For the ensemble model
            if(ensemble_acc_score>acc_score):
                print("Ensemble has Better Accuracy")

            # Return all scores if needed
            return {
                'Best Model': {
                    'Accuracy': acc_score * 100,
                    'Precision': precision * 100,
                    'Recall': recall * 100,
                    'F1 Score': f1 * 100,
                    'Confusion Matrix': confusion_best_model
                },
                'Ensemble Model': {
                    'Accuracy': ensemble_acc_score * 100,
                    'Precision': ensemble_precision * 100,
                    'Recall': ensemble_recall * 100,
                    'F1 Score': ensemble_f1 * 100,
                    'Confusion Matrix': confusion_ensemble
                }
            }

        except Exception as e:
            raise CustomException(e, sys)