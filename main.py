from text_classification.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline

from text_classification.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from text_classification.logging import logging
from text_classification.exceptions.exceptions import ClassificationException
import sys



STAGE_NAME = "Data Ingestion stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.initiate_data_ingestion()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        raise ClassificationException(e,sys)

STAGE_NAME = "Data Validation stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataValidationTrainingPipeline()
   data_ingestion.initiate_data_validation()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        raise ClassificationException(e,sys)