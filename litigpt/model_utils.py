"""
Shared utilities for model loading, constants, and metadata.

Centralizes model/tokenizer loading logic that was previously duplicated
across generator.py, trainer.py, gradio_app.py, and ollama.py.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logger = logging.getLogger(__name__)

DEFAULT_USERNAME = "anonimo"
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _detect_compute_dtype() -> Tuple[torch.dtype, bool]:
    """Detect optimal compute dtype based on GPU capabilities."""
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    return compute_dtype, use_bf16


def load_model_and_tokenizer(
    base_model: str,
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = True,
    for_training: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.dtype]:
    """
    Load a model and tokenizer with optional quantization and LoRA adapters.

    Args:
        base_model: HuggingFace model identifier or local path.
        adapter_path: Path to LoRA adapter directory. If None, loads base model only.
        load_in_4bit: Whether to use 4-bit quantization via BitsAndBytes.
        for_training: If True, prepares model for k-bit training and sets
                      padding_side="right". If False, loads adapter and calls eval().

    Returns:
        (model, tokenizer, compute_dtype) tuple.
    """
    compute_dtype, use_bf16 = _detect_compute_dtype()
    logger.info("Compute dtype: %s", "bfloat16" if use_bf16 else "float16")

    # Quantization config
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    # Load base model
    logger.info("Loading model: %s", base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )

    if for_training:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    elif adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
    else:
        model.eval()

    # Load tokenizer — prefer adapter_path (it may store a modified tokenizer)
    tokenizer_source = adapter_path if adapter_path and not for_training else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.pad_token = tokenizer.eos_token
    if for_training:
        tokenizer.padding_side = "right"

    logger.info("Model loaded successfully")
    return model, tokenizer, compute_dtype


def load_user_metadata(processed_dir: str) -> List[str]:
    """
    Load the list of available usernames from users_metadata.json.

    Returns an empty list if the file is missing or malformed.
    """
    metadata_path = Path(processed_dir) / "users_metadata.json"
    if not metadata_path.exists():
        return []
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        users = metadata.get("users", [])
        logger.info("Loaded %d users: %s", len(users), users)
        return users
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to read user metadata from %s: %s", metadata_path, e)
        return []
