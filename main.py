from src.diabeties import logger
from src.diabeties.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.diabeties.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from src.diabeties.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.diabeties.pipeline.model_trainer_pipeline import ModelTrainerTrainingPipeline
from src.diabeties.pipeline.model_evaluation_pipeline import ModelEvaluationTrainingPipeline

try:
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.initiate_data_ingestion()
except Exception as e:
        logger.exception(e)
        raise e

try:
   data_validation = DataValidationTrainingPipeline()
   data_validation.initiate_data_validation()
except Exception as e:
        logger.exception(e)
        raise e

try:
   data_ingestion = DataTransformationTrainingPipeline()
   data_ingestion.initiate_data_transformation()
except Exception as e:
        logger.exception(e)
        raise e
try:
   model_trainer = ModelTrainerTrainingPipeline()
   model_trainer.initiate_model_training()
except Exception as e:
        logger.exception(e)
        raise e

try:
   model_evaluation = ModelEvaluationTrainingPipeline()
   model_evaluation.initiate_model_evaluation()
except Exception as e:
        logger.exception(e)
        raise e