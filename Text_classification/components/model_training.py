# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# Evaluation
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,f1_score
from sklearn.model_selection import train_test_split

# Other
import numpy as np
import pandas as pd
# import logging

from text_classification.entity.config_entity import *
from text_classification.exceptions.exceptions import ClassificationException
from text_classification.logging import logging
import sys
import logging

import os
import joblib

class ModelTrainer:
    def __init__(self, config=ModelTrainerConfig):
        try:
            self.config = config
        except ClassificationException as e:
            raise(e, sys)
        
    def load_data(self):
        logging.info("loading data")
        train_df = pd.read_csv(self.config.train_data)
        test_df = pd.read_csv(self.config.test_data)

        X_train = train_df.iloc[:, :-1]
        y_train = train_df.iloc[:, -1]
        X_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1]

        return X_train, X_test, y_train, y_test

    def save_object(self, file_path, obj):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
        logging.info(f"Saved object to {file_path}")

    def model_training(self):
        try:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "SVC": SVC(probability=True),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "Naive Bayes": GaussianNB()
            }

            X_train, X_test, y_train, y_test = self.load_data()

            results = {}
            best_accuracy = 0
            best_model_name = None
            best_model = None

            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    results[name] = {
                        "Accuracy": acc,
                        "F1 Score": f1,
                        "Report": classification_report(y_test, y_pred, output_dict=False)
                    }
                    print(f"✅ {name} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
                    print(f" {name}Classification Report : {classification_report(y_test, y_pred, output_dict=False)} ")

                    # Update best model if current is better
                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_model_name = name
                        best_model = model

                except Exception as e:
                    print(f"❌ {name} failed: {e}")

            if best_model is not None:
                save_path = os.path.join("final_model", f"{best_model_name}_best_model.pkl")
                self.save_object(save_path, best_model)
                logging.info(f"Best model '{best_model_name}' saved at {save_path}")
                print(f"\nBest model saved: {best_model_name} with accuracy: {best_accuracy:.4f}")

            return results

        except ClassificationException as e:
            raise(e, sys)
