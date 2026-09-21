"""
Configuration module using Pydantic models.

Loads settings from config.yaml with validation, defaults, and type safety.
"""

import logging
from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    submission_filename: str = "litigi_submissions.parquet"
    comments_filename: str = "litigi_comments.parquet"
    processed_dir: str = "data/processed"
    training_dir: str = "data/training"
    target_usernames: List[str] = []
    # Leave target_usernames empty and set top_n_users to pick the N most
    # prolific authors automatically. An explicit target_usernames wins.
    top_n_users: int = 0
    # Cap pairs per user so one prolific author can't dominate the mix.
    # 0 disables the cap.
    max_pairs_per_user: int = 0
    # Never treat these as trainable personas.
    exclude_authors: List[str] = ["[deleted]", "[removed]", "AutoModerator"]
    # Drop the quoted parent text ("> ...") from a user's own reply. The parent
    # is already supplied as context, so keeping it teaches the model to copy.
    strip_quoted_text: bool = True
    min_comment_length: int = 10
    max_comment_length: int = 512
    min_score: int = 1


class ModelConfig(BaseModel):
    # Matches config.default.yaml. These defaults are not decoration: with no
    # config.yaml present, from_yaml() returns this model as-is, so anything
    # stale here is what an unconfigured run actually trains with. This used
    # to be meta-llama/Llama-3.1-8B-Instruct, a gated repo nothing in the
    # project trains against.
    base_model: str = "microsoft/phi-3-mini-4k-instruct"
    output_dir: str = "models/reddit_bot_lora"


class TrainingConfig(BaseModel):
    # 2 epochs: the top-30 run converged at roughly two and the third bought
    # 0.0014 of eval loss for an hour of rented GPU.
    num_epochs: int = 2
    # 4 x 8 holds the effective batch at 32 while halving peak activation
    # memory, which max_seq_length 1024 needs.
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    # Halved alongside the target-module fix, which tripled trainable
    # parameters.
    learning_rate: float = 1e-4
    # Loss is masked to the reply, so an example whose prompt fills the window
    # has nothing to learn from. At 512 that was 4.5% of the set.
    max_seq_length: int = 1024
    warmup_ratio: float = 0.05
    train_ratio: float = 0.9
    # Batch similar-length sequences together. Padding waste on this dataset
    # is ~103% without it, which roughly doubles cloud GPU cost.
    group_by_length: bool = True
    # Evaluation/checkpoint cadence. Defaults suit a small set; on a large
    # val split a low eval_steps dominates runtime (5,988 val examples at
    # batch 8 is 749 batches per evaluation).
    eval_steps: int = 50
    save_steps: int = 100
    logging_steps: int = 10
    save_total_limit: int = 3
    # Stop when eval_loss has not improved by more than the threshold for this
    # many consecutive evaluations. 0 disables it. The threshold matters as
    # much as the patience: improvements in the fourth decimal place are not
    # progress, but without a floor they keep the counter reset forever.
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.005


class LoraConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Empty means "detect from the loaded model", which is the right default:
    # projection names are architecture-specific (phi-3 fuses q/k/v into
    # qkv_proj), and a list that matches nothing trains silently on a fraction
    # of the network. Set explicitly only to adapt a deliberate subset.
    target_modules: List[str] = []


class InferenceConfig(BaseModel):
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1


class BotConfig(BaseModel):
    subreddit: str = "test"
    available_users: List[str] = []
    trigger_keywords: List[str] = []
    reply_probability: float = 0.2
    min_score_threshold: int = 1
    cooldown_seconds: int = 120
    max_context_depth: int = 3
    bot_username: str = "your_bot_username"


class UserClassificationConfig(BaseModel):
    strategy: str = "random"  # "random" or "keyword"
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
        logger.info("Created: %s", directory)
