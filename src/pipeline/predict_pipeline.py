import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import joblib

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model, feature_names = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)

            # Ensure only the expected features are passed to the model
            features = features[feature_names]
            data_scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e , sys)


class CustomData:
    def __init__(self,
        BMI: int,
        Dia_BP: int,
        OGTT: int,
        PCOS: int,
        Prediabetes: int):
        self.BMI = BMI
        self.Dia_BP = Dia_BP
        self.OGTT = OGTT
        self.PCOS = PCOS
        self.Prediabetes = Prediabetes
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "BMI": [self.BMI],
                "Dia_BP": [self.Dia_BP],
                "OGTT": [self.OGTT],
                "PCOS": [self.PCOS],
                "Prediabetes": [self.Prediabetes]
            }

            return pd.DataFrame(custom_data_input_dict, columns=["BMI", "Dia_BP", "OGTT", "PCOS", "Prediabetes"])
        
        except Exception as e:
            raise CustomException(e, sys)
