"""
Shared utilities for model loading, constants, and metadata.

Centralizes model/tokenizer loading logic that was previously duplicated
across generator.py, trainer.py, gradio_app.py, and ollama.py.
"""

import json
import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

DEFAULT_USERNAME = "anonimo"
# The base every config in this repo actually trains against. It used to be
# meta-llama/Llama-3.1-8B-Instruct, which is a gated repo: anyone who launched
# an interface without --base-model got an auth failure from Hugging Face
# rather than the adapter they just trained.
DEFAULT_BASE_MODEL = "microsoft/phi-3-mini-4k-instruct"


def _detect_compute_dtype() -> tuple[torch.dtype, bool]:
    """Detect optimal compute dtype based on GPU capabilities."""
    if not torch.cuda.is_available():
        # On CPU, float16 is the wrong answer: PyTorch's CPU kernels cover it
        # patchily and common ops raise "not implemented for 'Half'". float32
        # would be safest, but a 3.8B model needs ~15GB that way, which does
        # not fit on a typical laptop. bfloat16 is the workable middle at
        # ~7.6GB and has broad CPU coverage.
        logger.info("No CUDA device; using bfloat16 on CPU")
        return torch.bfloat16, False

    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    return compute_dtype, use_bf16


def _estimate_weight_bytes(base_model: str) -> int | None:
    """
    Size of a model's weight files, if it is already in the HuggingFace cache.

    Returns None when the size cannot be known — a local path, a model not yet
    downloaded, or no hub cache. Callers treat None as "cannot check".
    """
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
    except Exception:
        return None

    for repo in cache.repos:
        if repo.repo_id != base_model:
            continue
        sizes = [
            sum(f.size_on_disk for f in rev.files
                if f.file_name.endswith((".safetensors", ".bin")))
            for rev in repo.revisions
        ]
        largest = max(sizes, default=0)
        return largest or None
    return None


def _check_cpu_headroom(base_model: str) -> None:
    """
    Refuse a CPU load that cannot fit in RAM, before spending minutes on it.

    Without this the load appears to succeed: accelerate quietly offloads the
    overflow to disk, those parameters stay on the meta device, and any LoRA
    weights written to them are discarded as a no-op. The result is a model
    that runs and answers with the adapter missing from much of the network,
    which is far worse than a refusal.
    """
    needed = _estimate_weight_bytes(base_model)
    if needed is None:
        return

    try:
        import psutil
        available = psutil.virtual_memory().available
    except Exception:
        return

    gb = 1024 ** 3
    # Weights plus room for activations, KV cache and the tokenizer.
    required = needed * 1.15
    if available >= required:
        return

    raise RuntimeError(
        f"Not enough RAM to load {base_model} on CPU.\n"
        f"  weights:   {needed / gb:.1f} GB\n"
        f"  need:      {required / gb:.1f} GB (weights + activations)\n"
        f"  available: {available / gb:.1f} GB\n\n"
        "Close other applications to free memory, or run the model on a "
        "machine with a CUDA GPU and reach it over the network "
        "(launch_chat.py ... --share).\n"
        "The blind evaluation's --oracle mode needs no model at all and "
        "runs fine here."
    )


def load_model_and_tokenizer(
    base_model: str,
    adapter_path: str | None = None,
    load_in_4bit: bool = True,
    for_training: bool = False,
    trust_remote_code: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, torch.dtype]:
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
    compute_dtype, _use_bf16 = _detect_compute_dtype()
    logger.info("Compute dtype: %s", str(compute_dtype).replace("torch.", ""))

    # bitsandbytes 4-bit kernels are CUDA-only. Asking for them on a CPU-only
    # box fails deep inside the loader with an unhelpful error, so degrade to
    # an unquantized load and say so.
    if load_in_4bit and not torch.cuda.is_available():
        logger.warning(
            "4-bit quantization needs a CUDA GPU; loading unquantized on CPU "
            "instead. Expect ~7.6GB of RAM for a 3.8B model and slow "
            "generation — lower max_new_tokens to keep replies bearable."
        )
        load_in_4bit = False

    # Quantization config
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    on_cpu = not torch.cuda.is_available()
    if on_cpu:
        _check_cpu_headroom(base_model)

    # device_map="auto" hands the load to accelerate. On a GPU that is what we
    # want. On CPU it is actively harmful: anything that does not fit is
    # offloaded to disk and left on the meta device, adapter weights written to
    # those layers are silently dropped, and PEFT's offload fix-up then fails
    # with a KeyError about a mangled module path. Loading straight to CPU
    # keeps every parameter real, and the headroom check above is what decides
    # whether it fits.
    device_map = None if on_cpu else "auto"

    # Load base model
    logger.info("Loading model: %s", base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        dtype=compute_dtype,
        # Default False so transformers uses its own maintained implementation.
        # A model repo's bundled modeling_*.py is frozen at whatever transformers
        # API existed when it was uploaded: phi-3's calls the DynamicCache
        # attribute `seen_tokens`, removed in newer transformers, and generation
        # dies with "'DynamicCache' object has no attribute 'seen_tokens'".
        # The native implementation tracks the current cache API. Pass True only
        # for an architecture transformers does not support natively.
        trust_remote_code=trust_remote_code,
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


def load_user_metadata(processed_dir: str) -> list[str]:
    """
    Load the list of available usernames from users_metadata.json.

    Returns an empty list if the file is missing or malformed.
    """
    metadata_path = Path(processed_dir) / "users_metadata.json"
    if not metadata_path.exists():
        return []
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        users = metadata.get("users", [])
        logger.info("Loaded %d users: %s", len(users), users)
        return users
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to read user metadata from %s: %s", metadata_path, e)
        return []
