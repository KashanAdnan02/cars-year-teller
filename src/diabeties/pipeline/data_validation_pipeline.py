from src.diabeties.config.configuration import ConfigurationManager
from src.diabeties.components.data_validation import DataValiadtion
from src.diabeties import logger

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_validation(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValiadtion(config=data_validation_config)
        data_validation.validate_all_columns()

if __name__ == '__main__':
    try:
        obj = DataValidationTrainingPipeline()
        obj.initiate_data_validation()
    except Exception as e:
        logger.exception(e)
        raise e