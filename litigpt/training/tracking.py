"""
Module 8: MLflow Experiment Tracking
Track training experiments, model versions, and metrics
"""

import mlflow
import mlflow.pytorch
from pathlib import Path
import yaml
from typing import Dict, Any
import pandas as pd
from datetime import datetime

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
        
        print(f"MLflow tracking initialized: {tracking_uri}")
        print(f"Experiment: {experiment_name}")
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None):
        """Start a new MLflow run"""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        mlflow.start_run(run_name=run_name, tags=tags)
        print(f"Started MLflow run: {run_name}")
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
        print("MLflow run ended")
    
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
        
        print(f"\nTop {n_best} runs by {metric}:")
        print("-" * 80)
        
        results = []
        for i, run in enumerate(runs, 1):
            metrics = run.data.metrics
            params = run.data.params
            
            print(f"{i}. Run ID: {run.info.run_id}")
            print(f"   {metric}: {metrics.get(metric, 'N/A')}")
            print(f"   Learning rate: {params.get('training.learning_rate', 'N/A')}")
            print(f"   Epochs: {params.get('training.num_epochs', 'N/A')}")
            print()
            
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
        
        print(f"Loading best model from run: {best_run_id}")
        return mlflow.pytorch.load_model(model_uri)

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, bot_inference):
        self.bot = bot_inference
    
    def evaluate_perplexity(self, test_data: list) -> float:
        """Calculate perplexity on test set"""
        import torch
        import numpy as np
        
        total_loss = 0
        total_tokens = 0
        
        for example in test_data:
            context = example.get('context', '')
            target = example.get('response', '')
            
            # Get model's predicted probability
            # This is simplified - full implementation would use model.forward()
            # to get actual log probabilities
            pass
        
        perplexity = np.exp(total_loss / total_tokens)
        return perplexity
    
    def evaluate_response_quality(self, test_contexts: list) -> Dict[str, float]:
        """Evaluate response quality metrics"""
        
        from nltk.translate.bleu_score import sentence_bleu
        from rouge_score import rouge_scorer
        
        bleu_scores = []
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        
        for example in test_contexts:
            context = example['context']
            reference = example['response']
            
            # Generate prediction
            prediction = self.bot.generate_response(context)
            
            # BLEU score
            bleu = sentence_bleu([reference.split()], prediction.split())
            bleu_scores.append(bleu)
            
            # ROUGE scores
            rouge = scorer.score(reference, prediction)
            for key in rouge_scores:
                rouge_scores[key].append(rouge[key].fmeasure)
        
        return {
            'avg_bleu': sum(bleu_scores) / len(bleu_scores),
            'avg_rouge1': sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1']),
            'avg_rouge2': sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2']),
            'avg_rougeL': sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL'])
        }
    
    def evaluate_style_consistency(self, test_contexts: list, 
                                   user_samples: list) -> Dict[str, float]:
        """Evaluate if generated text matches user's style"""
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Generate responses
        generated = [
            self.bot.generate_response(ex['context']) 
            for ex in test_contexts
        ]
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=1000)
        
        # Fit on user samples
        user_vectors = vectorizer.fit_transform(user_samples)
        generated_vectors = vectorizer.transform(generated)
        
        # Calculate similarity
        similarities = cosine_similarity(generated_vectors, user_vectors)
        avg_similarity = similarities.mean()
        
        return {
            'style_similarity': avg_similarity,
            'min_similarity': similarities.min(),
            'max_similarity': similarities.max()
        }

# Integration with training module
def integrate_mlflow_with_trainer(trainer, tracker: MLflowTracker, config: dict):
    """
    Integrate MLflow tracking with the training process.
    Dead code / design example - the trainer's __main__ block already does
    this inline. If you want callbacks during training, wire MLflowCallback
    into the HuggingFace Trainer via trainer.add_callback().
    """
    
    # Start MLflow run
    run_name = f"reddit_bot_{config['data']['target_username']}"
    tracker.start_run(run_name=run_name, tags={
        'model': config['model']['base_model'],
        'user': config['data']['target_username']
    })
    
    # Log config
    tracker.log_config(config)
    
    # Add callback for logging metrics during training
    class MLflowCallback:
        def __init__(self, tracker):
            self.tracker = tracker
        
        def on_epoch_end(self, epoch, logs):
            self.tracker.log_training_metrics(
                epoch=epoch,
                train_loss=logs.get('train_loss'),
                val_loss=logs.get('val_loss'),
                learning_rate=logs.get('learning_rate')
            )
    
    return MLflowCallback(tracker)

if __name__ == "__main__":
    # Example usage
    tracker = MLflowTracker(
        experiment_name="reddit-chatbot",
        tracking_uri="file:./mlruns"
    )
    
    # Start a run
    tracker.start_run(run_name="test_run", tags={'model': 'llama-3.1-8b'})
    
    # Log parameters
    config = {
        'training': {
            'learning_rate': 2e-4,
            'num_epochs': 3,
            'batch_size': 4
        }
    }
    tracker.log_config(config)
    
    # Log metrics (simulated)
    for epoch in range(3):
        tracker.log_training_metrics(
            epoch=epoch,
            train_loss=0.5 - epoch * 0.1,
            val_loss=0.6 - epoch * 0.1
        )
    
    # End run
    tracker.end_run()
    
    # Compare runs
    tracker.compare_runs(metric="val_loss", n_best=3)
