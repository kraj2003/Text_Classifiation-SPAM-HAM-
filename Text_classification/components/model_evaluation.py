import pandas as pd
from urllib.parse import urlparse
from mlflow.models import infer_signature
import joblib
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import os
import sys
import mlflow

from text_classification.entity.config_entity import ModelEvaluationConfig
from text_classification.exceptions.exceptions import ClassificationException
from text_classification.logging import logging
import logging
from sklearn.metrics import accuracy_score,classification_report
import numpy as np
from text_classification.utils.common import *

os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/kraj2003/Text_Classifiation-SPAM-HAM-.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="kraj2003"
os.environ["MLFLOW_TRACKING_PASSWORD"]="e14d9832e6aa6cdfedb20a53788cc42efe4bcc39"

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        try:
            self.config = config
        except ClassificationException as e:
            raise e

    def log_into_mlflow(self):
        try:
            test_data = pd.read_csv(self.config.test_data_path)
            model = joblib.load(self.config.model_path)


            test_x = test_data.iloc[:, :-1]
            test_y = test_data.iloc[:, -1]

            signature=infer_signature(test_x,test_y)

            y_pred = model.predict(test_x)
            acc = accuracy_score(test_y, y_pred)
            report = classification_report(test_y, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(test_y, y_pred).tolist()

            results = {
                    "accuracy": acc,
                    "confusion_matrix": conf_matrix,
                    "classification_report": report
                }
            # Save locally as well
            os.makedirs(self.config.metric_file_name, exist_ok=True)
            save_json(Path(self.config.metric_file_name )/ "evaluation_metrics.json", results)

            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                mlflow.log_metric('accuracy',acc)
                mlflow.log_text(str(report), "classification_report.json")
                mlflow.log_text(str(conf_matrix), "confusion_matrix.json")
                # mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="MushroomClassifierRF")
                if tracking_url!='file':
                    mlflow.sklearn.log_model(model,"model",registered_model_name="Best Model")

                else:
                    mlflow.sklearn.log_model(model,"model",signature=signature)

                print("✅ Metrics and model logged to MLflow")
                print(f"📊 Accuracy: {acc:.4f}")
                print(f"Confusion Metrix {conf_matrix}")

            return results
        except ClassificationException as e:
            raise(e,sys)
    
