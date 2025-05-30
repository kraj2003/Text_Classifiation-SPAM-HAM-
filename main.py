from text_classification.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline

from text_classification.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from text_classification.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from text_classification.pipeline.model_trainer_pipeline import ModelTrainingPipeline
from text_classification.pipeline.model_evaluation import ModelEvalPipeline
# from text_classification.logging import logging
import logging
from text_classification.exceptions.exceptions import ClassificationException
import sys



# STAGE_NAME = "Data Ingestion stage"
# try:
#    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_ingestion = DataIngestionTrainingPipeline()
#    data_ingestion.initiate_data_ingestion()
#    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         raise ClassificationException(e,sys)

# STAGE_NAME = "Data Validation stage"
# try:
#    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_validation = DataValidationTrainingPipeline()
#    data_validation.initiate_data_validation()
#    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         raise ClassificationException(e,sys)

# STAGE_NAME = "Data Transformation stage"
# try:
#    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_transformation = DataTransformationTrainingPipeline()
#    data_transformation.initiate_data_transformation()
#    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         raise ClassificationException(e,sys)

# STAGE_NAME = "Model Training stage"
# try:
#    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    model_trainer = ModelTrainingPipeline()
#    model_trainer.initiate_model_training()
#    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         raise ClassificationException(e,sys)

STAGE_NAME = "Model Evaluation stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_trainer = ModelEvalPipeline()
   model_trainer.initiate_model_eval()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        raise ClassificationException(e,sys)