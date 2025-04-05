from text_classification.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig
from text_classification.utils.common import *
from text_classification.constants import *
from text_classification.exceptions.exceptions import ClassificationException
from text_classification.logging import logging

class ConfigurationManager:
    def __init__(self,
                 config_filepath= CONFIG_FILE_PATH,
                 schema_filepath = SCHEMA_FILE_PATH):
        self.config=read_yaml(config_filepath)
        self.schema=read_yaml(schema_filepath)
        create_directories([self.config["artifacts_root"]])
    def get_data_ingestion_config(self)->DataIngestionConfig:
        config=self.config["data_ingestion"]

        create_directories([config["root_dir"]])

        data_ingestion_config=DataIngestionConfig(
            root_dir=config["root_dir"],
            source_file_path=config["source_file_path"]
        )
        return data_ingestion_config
    
    def get_data_validation_config(self)->DataValidationConfig:
        config=self.config["data_validation"] 
        schema=self.schema["COLUMNS"]
        create_directories([config["root_dir"]])

        data_validation_config=DataValidationConfig(
            root_dir=config["root_dir"],
            STATUS_FILE=config["STATUS_FILE"],
            data_dir=config["data_dir"],
            all_schema=schema
            )
        return data_validation_config
        

    def get_data_transformation_config(self)->DataTransformationConfig:

        config=self.config["data_transformation"]
        create_directories([config["root_dir"]])
        create_directories([config["final_model"]])

        data_transformation_config=DataTransformationConfig(
            root_dir=config["root_dir"],
            data_path=config["data_path"],
            final_model=config["final_model"],
            preprocessor=config["preprocessor"]
        )
        return data_transformation_config