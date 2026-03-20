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
        """Configure LoRA parameters"""

        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

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
    
    def format_chat_template(self, example, tokenizer):
        """Format examples using chat template"""
        if "messages" in example:
            # ChatML format
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # Raw text format
            text = example["text"]
        
        return {"text": text}
    
    def train(self,
              data_dir: str = "data/training",
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 2e-4,
              max_seq_length: int = 512,
              gradient_accumulation_steps: int = 4,
              lora_r: int = 16,
              lora_alpha: int = 32,
              lora_dropout: float = 0.05,
              lora_target_modules: list = None,
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
        
        # Format dataset
        dataset = dataset.map(
            lambda x: self.format_chat_template(x, tokenizer),
            remove_columns=dataset["train"].column_names
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
            warmup_ratio=0.05,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            fp16=not self.use_bf16,
            bf16=self.use_bf16,
            report_to=report_to,
            load_best_model_at_end=True,
            max_length=max_seq_length,
            dataset_text_field="text",
            packing=False,
        )

        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            processing_class=tokenizer,
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
            torch_dtype=torch.float16,
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