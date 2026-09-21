"""
Module 8: MLflow Experiment Tracking
Track training experiments, model versions, and metrics
"""

import logging

import mlflow
import mlflow.pytorch
from typing import Dict, Any
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class MLflowTracker:
    def __init__(self, 
                 experiment_name: str = "reddit-chatbot",
                 tracking_uri: str = "file:./mlruns"):
        """
        Initialize MLflow tracking
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: Where to store MLflow data (local or remote)
        """
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        
        logger.info("MLflow tracking initialized: %s", tracking_uri)
        logger.info("Experiment: %s", experiment_name)
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None):
        """Start a new MLflow run"""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        mlflow.start_run(run_name=run_name, tags=tags)
        logger.info("Started MLflow run: %s", run_name)
        return mlflow.active_run()
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration parameters"""
        
        # Flatten nested config
        flat_config = self._flatten_dict(config)
        
        for key, value in flat_config.items():
            mlflow.log_param(key, value)
    
    def log_dataset_stats(self, train_size: int, val_size: int, 
                         user_comments: int, avg_length: float):
        """Log dataset statistics"""
        mlflow.log_metric("train_size", train_size)
        mlflow.log_metric("val_size", val_size)
        mlflow.log_metric("user_comments", user_comments)
        mlflow.log_metric("avg_comment_length", avg_length)
    
    def log_training_metrics(self, epoch: int, train_loss: float, 
                           val_loss: float = None, learning_rate: float = None):
        """Log training metrics per epoch"""
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        
        if val_loss is not None:
            mlflow.log_metric("val_loss", val_loss, step=epoch)
        
        if learning_rate is not None:
            mlflow.log_metric("learning_rate", learning_rate, step=epoch)
    
    def log_evaluation_metrics(self, metrics: Dict[str, float]):
        """Log evaluation metrics"""
        for key, value in metrics.items():
            mlflow.log_metric(f"eval_{key}", value)
    
    def log_model(self, model_path: str, model_name: str = "reddit_bot"):
        """Log the trained model"""
        mlflow.log_artifact(model_path, artifact_path="model")
        
        # Register model in registry
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            model_name
        )
    
    def log_sample_outputs(self, samples: list):
        """Log sample model outputs for qualitative evaluation"""
        
        # Create DataFrame of samples
        df = pd.DataFrame(samples)
        
        # Save as CSV artifact
        output_path = "sample_outputs.csv"
        df.to_csv(output_path, index=False)
        mlflow.log_artifact(output_path)
    
    def end_run(self):
        """End the current MLflow run"""
        mlflow.end_run()
        logger.info("MLflow run ended")
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def compare_runs(self, metric: str = "val_loss", n_best: int = 5):
        """Compare runs and get best models"""
        
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} ASC"],
            max_results=n_best
        )
        
        logger.info("Top %d runs by %s:", n_best, metric)

        results = []
        for i, run in enumerate(runs, 1):
            metrics = run.data.metrics
            params = run.data.params

            logger.info(
                "%d. Run %s — %s: %s, lr: %s, epochs: %s",
                i, run.info.run_id,
                metric, metrics.get(metric, "N/A"),
                params.get("training.learning_rate", "N/A"),
                params.get("training.num_epochs", "N/A"),
            )
            
            results.append({
                'run_id': run.info.run_id,
                metric: metrics.get(metric),
                'learning_rate': params.get('training.learning_rate'),
                'epochs': params.get('training.num_epochs')
            })
        
        return results
    
    def load_best_model(self, metric: str = "val_loss"):
        """Load the best model based on metric"""
        
        best_runs = self.compare_runs(metric=metric, n_best=1)
        if not best_runs:
            raise ValueError("No runs found")
        
        best_run_id = best_runs[0]['run_id']
        model_uri = f"runs:/{best_run_id}/model"
        
        logger.info("Loading best model from run: %s", best_run_id)
        return mlflow.pytorch.load_model(model_uri)
