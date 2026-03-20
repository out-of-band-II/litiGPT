"""
Configuration module using Pydantic models.

Loads settings from config.yaml with validation, defaults, and type safety.
"""

from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    submission_filename: str = "litigi_submissions.jsonl"
    comments_filename: str = "litigi_comments.parquet"
    processed_dir: str = "data/processed"
    training_dir: str = "data/training"
    target_usernames: List[str] = []
    min_comment_length: int = 10
    max_comment_length: int = 512
    min_score: int = 1


class ModelConfig(BaseModel):
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir: str = "models/reddit_bot_lora"


class TrainingConfig(BaseModel):
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    warmup_ratio: float = 0.05


class LoraConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


class InferenceConfig(BaseModel):
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1


class BotConfig(BaseModel):
    subreddit: str = "test"
    available_users: List[str] = []
    user_classifier_path: str = "models/user_classifier.pkl"
    trigger_keywords: List[str] = []
    reply_probability: float = 0.2
    min_score_threshold: int = 1
    cooldown_seconds: int = 120
    max_context_depth: int = 3
    bot_username: str = "your_bot_username"


class UserClassificationConfig(BaseModel):
    method: str = "tfidf"
    user_keywords: Dict[str, List[str]] = {}


class Config(BaseModel):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    lora: LoraConfig = LoraConfig()
    inference: InferenceConfig = InferenceConfig()
    bot: BotConfig = BotConfig()
    user_classification: UserClassificationConfig = UserClassificationConfig()

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Config":
        """Load config from a YAML file, falling back to defaults for missing keys."""
        config_path = Path(path)
        if not config_path.exists():
            return cls()
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)


def setup_project_structure():
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/processed",
        "data/training",
        "models",
        "logs",
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")
