import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.cars import logger
from src.cars.entity.config_entity import DataTransformationConfig
from src.cars.config.configuration import ConfigurationManager

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_spliting(self):
        """
        Loads car data, applies encoding to categorical features,
        performs train-test split, and saves the processed datasets.
        """
        try:
            # Load data
            data = pd.read_csv(self.config.data_path)
            config_manager = ConfigurationManager()
            schema = config_manager.schema
            target_column = schema.TARGET_COLUMN.name  # Engine Capacity

            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")

            X = data.drop(columns=[target_column])
            y = data[target_column]

            # List of categorical columns to encode
            categorical_cols = [
                "nam", "Price", "Millage", "Fuel", "Transmission",
                "Province", "Color", "Assembly", "Body Type", "Features", "Owner nam"
            ]

            for col in categorical_cols:
                if col in X.columns:
                    le = LabelEncoder()
                    X[col] = X[col].fillna("Unknown")
                    X[col] = le.fit_transform(X[col].astype(str))
                else:
                    logger.warning(f"Column '{col}' not found in dataset → skipping encoding")

            # Train-test split
            train_x, test_x, train_y, test_y = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=None  # Engine Capacity likely continuous, so no stratify
            )

            # Combine X and y for saving
            train = pd.concat([train_x.reset_index(drop=True), train_y.reset_index(drop=True)], axis=1)
            test = pd.concat([test_x.reset_index(drop=True), test_y.reset_index(drop=True)], axis=1)

            train.rename(columns={train.columns[-1]: target_column}, inplace=True)
            test.rename(columns={test.columns[-1]: target_column}, inplace=True)

            # Save processed datasets
            os.makedirs(self.config.root_dir, exist_ok=True)
            train_path = os.path.join(self.config.root_dir, "train.csv")
            test_path = os.path.join(self.config.root_dir, "test.csv")

            train.to_csv(train_path, index=False)
            test.to_csv(test_path, index=False)

            logger.info("Car data transformation completed successfully")
        except Exception as e:
            logger.exception(f"Error in data transformation: {str(e)}")
            raise
