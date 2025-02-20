import pandas as pd
import numpy 
import os
import sys
from pathlib import Path
from typing import List
from sklearn.model_selection import train_test_split
# from dotenv import load_dotenv
from text_classification.entity.config_entity import DataIngestionConfig
from text_classification.logging import logging
from text_classification.exceptions.exceptions import ClassificationException

class DataIngestion:
    def __init__(self,config=DataIngestionConfig):
        self.config=config

    def read_data(self):
        try: 
            data=pd.read_csv('research\SMSSpamCollection.txt',sep='\t',names=["label","message"])
            data.to_csv(os.path.join(self.config.root_dir, "data.csv"),index = False)
        except Exception as e:
            raise ClassificationException(e,sys)