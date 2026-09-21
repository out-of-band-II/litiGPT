"""
Module 3: Model Training
Fine-tune model using QLoRA
"""

import logging
import os

import torch
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

from litigpt.model_utils import load_model_and_tokenizer as _load_model, DEFAULT_BASE_MODEL

logger = logging.getLogger(__name__)

# Heads and embeddings are linear too, but adapting them is a different
# decision with a large parameter cost, so auto-detection leaves them out.
_NON_BLOCK_LINEARS = {"lm_head", "score", "classifier", "embed_out"}

# bitsandbytes replaces nn.Linear during a 4-bit load, so match on class name
# rather than isinstance(module, nn.Linear), which would find nothing under
# QLoRA -- exactly the case this project trains in.
_LINEAR_CLASSES = {"Linear", "Linear4bit", "Linear8bitLt", "LinearNF4"}


def _leaf_names(model) -> set:
    """Every module's final path segment, which is what PEFT matches on."""
    return {name.rsplit(".", 1)[-1] for name, _ in model.named_modules() if name}


def discover_target_modules(model) -> list:
    """
    Find the linear projections inside the transformer blocks.

    Returns names suitable for LoraConfig.target_modules. Deriving them from
    the model rather than hard-coding keeps this correct across architectures
    that fuse their projections: Llama exposes q/k/v_proj and gate/up_proj,
    while phi-3 fuses the same tensors into qkv_proj and gate_up_proj, so a
    Llama-shaped list matches nothing on phi-3.
    """
    names = set()
    for name, module in model.named_modules():
        if module.__class__.__name__ not in _LINEAR_CLASSES:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in _NON_BLOCK_LINEARS:
            continue
        names.add(leaf)

    if not names:
        raise RuntimeError(
            "Found no linear projections to adapt. The model may have loaded "
            "with an unexpected layer implementation; pass target_modules "
            "explicitly."
        )
    return sorted(names)


def validate_target_modules(model, target_modules) -> None:
    """
    Raise unless every requested module name exists in the model.

    A name that matches nothing is not a warning-level problem. PEFT adapts
    the names it recognises, ignores the rest without comment, and writes the
    full requested list into adapter_config.json regardless -- so the config
    on disk claims seven modules while the weights hold two, and nothing in
    the logs or the loss curve reveals it.
    """
    present = _leaf_names(model)
    missing = [m for m in target_modules if m not in present]
    if not missing:
        return

    matched = [m for m in target_modules if m in present]
    available = discover_target_modules(model)
    arch = getattr(getattr(model, "config", None), "model_type", "unknown")

    raise ValueError(
        f"LoRA target_modules do not match this model ({arch}).\n"
        f"  requested:   {', '.join(target_modules)}\n"
        f"  not found:   {', '.join(missing)}\n"
        f"  would match: {', '.join(matched) or '(nothing)'}\n"
        f"  available:   {', '.join(available)}\n\n"
        "Set lora.target_modules in the config to the available names, or "
        "leave it empty to detect them automatically. Training with this "
        "list would silently adapt only part of the network."
    )


class RedditModelTrainer:
    def __init__(self,
                 model_name: str = DEFAULT_BASE_MODEL,
                 output_dir: str = "models/reddit_bot"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model_and_tokenizer(self):
        """Load model with 4-bit quantization for QLoRA"""
        model, tokenizer, compute_dtype = _load_model(
            base_model=self.model_name,
            load_in_4bit=True,
            for_training=True,
        )
        self.compute_dtype = compute_dtype
        self.use_bf16 = compute_dtype == torch.bfloat16
        return model, tokenizer
    
    def setup_lora(self, model, r: int = 16, lora_alpha: int = 32,
                   lora_dropout: float = 0.05, target_modules: list = None):
        """
        Configure LoRA parameters.

        With target_modules left as None the projections are discovered from
        the loaded model, which is the safe default: hard-coded name lists are
        architecture-specific and fail silently on anything else.
        """

        if not target_modules:
            target_modules = discover_target_modules(model)
            logger.info(
                "Auto-detected target modules: %s", ", ".join(target_modules)
            )

        # Refuse to train on a target list that does not match the model.
        # PEFT adapts whatever it finds and says nothing about the rest, so a
        # list written for another architecture produces a run that looks
        # entirely healthy -- loss falls, checkpoints save -- while most of the
        # network is untouched. This project lost a full 3-epoch run that way.
        validate_target_modules(model, target_modules)

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model
    
    def prepare_dataset(self, data_dir: str = "data/training"):
        """Load and prepare dataset"""
        
        # Load datasets
        dataset = load_dataset(
            "json",
            data_files={
                "train": f"{data_dir}/train.jsonl",
                "validation": f"{data_dir}/val.jsonl"
            }
        )
        
        logger.info("Train samples: %d", len(dataset['train']))
        logger.info("Validation samples: %d", len(dataset['validation']))
        
        return dataset
    
    @staticmethod
    def split_prompt_completion(example):
        """
        Split a ChatML example into the prompt and the reply to be learned.

        This is what makes the loss mask work. Handed a single block of text,
        TRL scores every token in it -- the system prompt and the user's
        context included -- so the model is trained to produce the other
        side of the conversation as much as its own reply. Handed a
        prompt/completion pair, it masks the prompt and scores only the
        completion, which is the thing being learned.

        TRL applies the chat template itself for conversational
        prompt/completion data, so no templating happens here.
        """
        messages = example["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError(
                "Every training example must end with an assistant message; "
                f"this one ends with {messages[-1].get('role') if messages else 'nothing'}."
            )
        return {"prompt": messages[:-1], "completion": messages[-1:]}

    def format_chat_template(self, example, tokenizer):
        """Render a raw-text example. Retained for non-ChatML datasets."""
        return {"text": example["text"]}

    @staticmethod
    def drop_unlearnable(dataset, tokenizer, max_seq_length: int):
        """
        Remove examples whose reply cannot survive truncation.

        Sequences are truncated from the right at max_seq_length. When the
        prompt alone already fills it, the reply is cut away entirely and the
        example has no unmasked tokens left -- it contributes nothing to the
        loss while still costing a full forward and backward pass. Under the
        old whole-sequence loss these examples were merely lopsided; with the
        prompt masked they are empty.

        On this dataset at 512 tokens that was 4.5% of examples, and at 1024
        it is 0.6%. Dropping them is honest about the data; the alternative
        is to trim context from the left during preprocessing.
        """
        def keeps_reply(example):
            prompt_len = len(tokenizer.apply_chat_template(
                example["prompt"], tokenize=True, add_generation_prompt=True
            ))
            # Leave room for at least one real reply token.
            return prompt_len < max_seq_length - 1

        before = {k: len(v) for k, v in dataset.items()}
        dataset = dataset.filter(keeps_reply)
        for split, n_before in before.items():
            dropped = n_before - len(dataset[split])
            if dropped:
                logger.warning(
                    "%s: dropped %d of %d examples (%.1f%%) whose reply does "
                    "not fit within max_seq_length=%d",
                    split, dropped, n_before, 100 * dropped / n_before,
                    max_seq_length,
                )
        return dataset
    
    def train(self,
              data_dir: str = "data/training",
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 2e-4,
              max_seq_length: int = 512,
              gradient_accumulation_steps: int = 4,
              warmup_ratio: float = 0.05,
              group_by_length: bool = True,
              eval_steps: int = 50,
              save_steps: int = 100,
              logging_steps: int = 10,
              save_total_limit: int = 3,
              lora_r: int = 16,
              lora_alpha: int = 32,
              lora_dropout: float = 0.05,
              lora_target_modules: list = None,
              early_stopping_patience: int = 3,
              early_stopping_threshold: float = 0.005,
              report_to: str = "none"):
        """Train the model"""

        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()

        # Setup LoRA
        model = self.setup_lora(
            model, r=lora_r, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout, target_modules=lora_target_modules,
        )
        
        # Prepare dataset
        dataset = self.prepare_dataset(data_dir)

        # Format dataset. ChatML becomes prompt/completion so the prompt is
        # masked out of the loss; anything else stays a plain text field and
        # is scored whole, which is all that can be done without a role
        # boundary to split on.
        columns = dataset["train"].column_names
        conversational = "messages" in columns
        if conversational:
            dataset = dataset.map(
                self.split_prompt_completion, remove_columns=columns
            )
            logger.info("Dataset is conversational; loss masked to the reply only")
            dataset = self.drop_unlearnable(dataset, tokenizer, max_seq_length)
        else:
            dataset = dataset.map(
                lambda x: self.format_chat_template(x, tokenizer),
                remove_columns=columns,
            )
            logger.warning(
                "Dataset has no 'messages' column; training on whole sequences, "
                "which also teaches the model to write the user's turns."
            )


        # Training arguments (SFTConfig = TrainingArguments + SFT-specific params)
        training_args = SFTConfig(
            output_dir=self.output_dir,
            logging_dir=os.path.join(self.output_dir, "logs"),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=warmup_ratio,
            group_by_length=group_by_length,
            logging_steps=logging_steps,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            fp16=not self.use_bf16,
            bf16=self.use_bf16,
            report_to=report_to,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            max_length=max_seq_length,
            packing=False,
            # For prompt/completion data this makes TRL label the prompt
            # tokens -100 so they contribute nothing to the loss. It has no
            # meaning for a plain text field, hence the conditional.
            **(
                {"completion_only_loss": True}
                if conversational
                else {"dataset_text_field": "text"}
            ),
        )

        # Stop once eval_loss stops moving meaningfully. The top-30 run
        # converged at step 3500 and then spent 1552 more steps -- 31% of the
        # run, about an hour of rented GPU -- to improve eval_loss by 0.0014.
        # The threshold is what makes this work: without it, improvements in
        # the fourth decimal place keep resetting the patience counter.
        callbacks = []
        if early_stopping_patience and early_stopping_patience > 0:
            from transformers import EarlyStoppingCallback
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            ))
            logger.info(
                "Early stopping: patience %d evaluations, threshold %.4f",
                early_stopping_patience, early_stopping_threshold,
            )

        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            processing_class=tokenizer,
            callbacks=callbacks,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()

        # Save final model
        logger.info("Saving model to %s", self.output_dir)
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        
        return trainer
    
    def merge_and_save_full_model(self, adapter_path: str = None):
        """Merge LoRA adapters with base model and save"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        if adapter_path is None:
            adapter_path = self.output_dir

        logger.info("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map="auto"
        )

        logger.info("Loading LoRA adapters...")
        model = PeftModel.from_pretrained(model, adapter_path)

        logger.info("Merging...")
        model = model.merge_and_unload()

        output_path = f"{self.output_dir}_merged"
        logger.info("Saving merged model to %s", output_path)
        model.save_pretrained(output_path)

        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        tokenizer.save_pretrained(output_path)

        return output_path

if __name__ == "__main__":
    # Example usage with MLflow tracking
    from litigpt.config import Config
    from litigpt.training.tracking import MLflowTracker

    # Load config
    config = Config.from_yaml("config.yaml")

    # Initialize MLflow
    tracker = MLflowTracker(
        experiment_name="reddit-chatbot-training",
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
    )

    # Start run
    users_label = "_".join(config.data.target_usernames)
    run_name = f"train_{users_label}"
    tracker.start_run(run_name=run_name, tags={
        "model": config.model.base_model,
        "users": ",".join(config.data.target_usernames),
    })

    # Log config
    tracker.log_config(config.model_dump())

    # Initialize trainer
    trainer = RedditModelTrainer(
        model_name=config.model.base_model,
        output_dir=config.model.output_dir,
    )

    # Train with MLflow logging
    trainer.train(
        data_dir=config.data.training_dir,
        num_epochs=config.training.num_epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
    )

    # Log model
    tracker.log_model(config.model.output_dir, model_name="reddit_bot")

    # End run
    tracker.end_run()

    logger.info("View results at: %s", tracker.tracking_uri)

    # Optional: merge LoRA adapters into the base model weights to produce a
    # single standalone model (no adapter files). Useful for Ollama/vLLM
    # deployments that don't support PEFT adapters natively. Costs extra VRAM
    # and disk space; skip unless you need a self-contained model file.
    # trainer.merge_and_save_full_model()