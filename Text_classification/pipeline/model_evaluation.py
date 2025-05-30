from text_classification.config.configuration import ConfigurationManager
from text_classification.components.model_evaluation import ModelEvaluation
# from text_classification.logging import logger
import logging

STAGE_NAME = "Model Evaluation stage"

class ModelEvalPipeline:
    def __init__(self):
        pass

    def initiate_model_eval(self):
        config = ConfigurationManager()
        model_eval_config = config.get_model_evaluation_config()
        model_eval = ModelEvaluation(config=model_eval_config)
        model_eval.log_into_mlflow()

if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvalPipeline()
        obj.initiate_model_eval()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e