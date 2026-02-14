import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from pathlib import Path

from src.diabeties import logger
from src.diabeties.entity.config_entity import ModelEvaluationConfig
from src.diabeties.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, predicted):
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        return rmse, mae, r2

    def log_into_mlflow(self):
        try:
            # 1. Load test data (this is the transformed version with encoded columns)
            if not os.path.exists(self.config.test_data_path):
                raise FileNotFoundError(f"Test data not found: {self.config.test_data_path}")

            logger.info(f"Loading test data from: {self.config.test_data_path}")
            test_data = pd.read_csv(self.config.test_data_path)
            logger.info(f"Test data shape: {test_data.shape}")

            # 2. Load the trained model
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"Model file not found: {self.config.model_path}")

            logger.info(f"Loading model from: {self.config.model_path}")
            model = joblib.load(self.config.model_path)

            # 3. Separate features and target
            target_col = self.config.target_column  # should be 'diabetes'
            
            if target_col not in test_data.columns:
                raise ValueError(f"Target column '{target_col}' not found in test data")

            test_x = test_data.drop(columns=[target_col])
            test_y = test_data[target_col]

            logger.info(f"Predicting on {test_x.shape[0]} test samples")
            logger.info(f"Feature columns: {list(test_x.columns)}")

            # 4. Make predictions
            predicted_values = model.predict(test_x)

            # 5. Calculate metrics
            rmse, mae, r2 = self.eval_metrics(test_y, predicted_values)

            # 6. Prepare results
            scores = {
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2)
            }

            logger.info(f"Evaluation metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

            # 7. Save metrics locally
            metrics_path = Path(self.config.metric_file_name)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(path=metrics_path, data=scores)
            logger.info(f"Metrics saved to: {metrics_path}")

            # 8. MLflow setup
            os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/KashanAdnan02/diabeties.mlflow/"
            os.environ["MLFLOW_TRACKING_USERNAME"] = "KashanAdnan02"
            os.environ["MLFLOW_TRACKING_PASSWORD"] = "0b3c46a2082fda864bb06352494ef5f1abde7118"

            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(self.config.all_params)

                # Log metrics
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                # Log model
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        name="model",
                        registered_model_name="ElasticNetDiabetesModel"
                    )
                else:
                    mlflow.sklearn.log_model(model, "model")

                logger.info("Model and metrics successfully logged to MLflow")

        except Exception as e:
            logger.exception(f"Error during model evaluation: {str(e)}")
            raise