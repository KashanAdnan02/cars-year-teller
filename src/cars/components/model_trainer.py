import pandas as pd
import os
from src.cars import logger
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from src.cars.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        try:
            # Load train/test data
            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)

            target = self.config.target_column

            X_train = train_data.drop([target], axis=1)
            y_train = train_data[target]
            X_test = test_data.drop([target], axis=1)
            y_test = test_data[target]

            # Identify categorical columns (string/object type)
            categorical_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
            numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

            # Build a preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                    ('num', 'passthrough', numeric_cols)
                ]
            )

            # Create full pipeline: preprocessing + ElasticNet
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42))
            ])

            # Train model
            pipeline.fit(X_train, y_train)

            # Save pipeline (includes preprocessing + model)
            os.makedirs(self.config.root_dir, exist_ok=True)
            model_path = os.path.join(self.config.root_dir, self.config.model_name)
            joblib.dump(pipeline, model_path)

            logger.info(f"Model trained and saved at: {model_path}")

        except Exception as e:
            logger.exception(f"Error in model training: {str(e)}")
            raise
