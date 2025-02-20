from text_classification.entity.config_entity import DataValidationConfig
from text_classification.exceptions.exceptions import ClassificationException
from text_classification.logging import logging
from text_classification.utils.common import *
import sys
import pandas as pd

class DataValidation:
    def __init__(self, config=DataValidationConfig):
        try:
            self.config=config

        except Exception as e:
            raise ClassificationException(e,sys)
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ClassificationException(e,sys)
        
    def validate_column(self)->bool:
        try:
            validation_status=None

            data=pd.read_csv(self.config.data_dir)
            all_cols=list(data.columns)
            columns_name=self.config.all_schema.keys()
            datatype=self.config.all_schema.values()
            
            logging.info(f"Required Number of columns:{columns_name}")
            logging.info(f"Dataframe has columns:{len(data.columns)}")
            for col in all_cols :
                if col not in columns_name and type(col) != datatype:
                    validation_status=False
                    with open(self.config.STATUS_FILE,'w') as f:
                        f.write(f"Validation_status: {validation_status}")
                else:
                    validation_status=True
                    with open(self.config.STATUS_FILE,'w') as f:
                        f.write(f"Validation Status : {validation_status}")


            return validation_status
        
        except Exception as e:
            raise ClassificationException(e,sys)