import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.diabeties import logger
from src.diabeties.entity.config_entity import DataTransformationConfig
from src.diabeties.config.configuration import ConfigurationManager


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_spliting(self):
        """
        Loads data, applies binary encoding to gender and smoking_history,
        performs train-test split, and saves the processed datasets.
        """
        try:
            data = pd.read_csv(self.config.data_path)
            config_manager = ConfigurationManager()
            schema = config_manager.schema
            target_column = schema.TARGET_COLUMN.name

            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")

            X = data.drop(columns=[target_column])
            y = data[target_column]
            if 'gender' in X.columns:
                gender_map = {'Male': 0, 'Female': 1}
                X['gender'] = X['gender'].map(gender_map)
                
                missing_gender = X['gender'].isna().sum()
                if missing_gender > 0:
                    logger.warning(f"{missing_gender} rows have missing or unknown gender value → filled with 0")
                    X['gender'] = X['gender'].fillna(0) 
                    
                X['gender'] = X['gender'].astype('Int64')
            else:
                logger.warning("Column 'gender' not found in dataset")

            if 'smoking_history' in X.columns:
                def encode_smoking(x):
                    if pd.isna(x):
                        return 1  # treat missing as 'No Info'
                    x_str = str(x).strip().lower()
                    if x_str == 'never':
                        return 0
                    else:
                        return 1  # No Info, current, former, ever, not current, etc.

                X['smoking_history'] = X['smoking_history'].apply(encode_smoking).astype('Int64')
            train_x, test_x, train_y, test_y = train_test_split(
                X, y,
                test_size=0.20,
                random_state=42,
                stratify=y if y.nunique() <= 10 else None )

            train = pd.concat([train_x.reset_index(drop=True), train_y.reset_index(drop=True)], axis=1)
            test = pd.concat([test_x.reset_index(drop=True), test_y.reset_index(drop=True)], axis=1)

            train.rename(columns={train.columns[-1]: target_column}, inplace=True)
            test.rename(columns={test.columns[-1]: target_column}, inplace=True)

            os.makedirs(self.config.root_dir, exist_ok=True)

            train_path = os.path.join(self.config.root_dir, "train.csv")
            test_path = os.path.join(self.config.root_dir, "test.csv")

            train.to_csv(train_path, index=False)
            test.to_csv(test_path, index=False)

            logger.info("Data transformation completed successfully")
        except Exception as e:
            logger.exception(f"Error in data transformation: {str(e)}")
            raise