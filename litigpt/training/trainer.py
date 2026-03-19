"""
Module 3: Model Training
Fine-tune model using QLoRA
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import os

class RedditModelTrainer:
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 output_dir: str = "models/reddit_bot"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # BFloat16 requires Ampere (sm_80) or newer; Pascal/Turing must use fp16
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.compute_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        print(f"Compute dtype: {'bfloat16' if self.use_bf16 else 'float16'}")
        
    def load_model_and_tokenizer(self):
        """Load model with 4-bit quantization for QLoRA"""
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        # Load model
        print(f"Loading model: {self.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=self.compute_dtype,  # non-quantized tensors (embeds, norms, LoRA) match compute dtype
        )
        
        # Prepare for training
        model = prepare_model_for_kbit_training(model)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        return model, tokenizer
    
    def setup_lora(self, model):
        """Configure LoRA parameters"""
        
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA alpha
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
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
        
        print(f"Train samples: {len(dataset['train'])}")
        print(f"Validation samples: {len(dataset['validation'])}")
        
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
              report_to: str = "none"):
        """Train the model"""
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Setup LoRA
        model = self.setup_lora(model)
        
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
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
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
        print("Starting training...")
        trainer.train()
        
        # Save final model
        print(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        
        return trainer
    
    def merge_and_save_full_model(self, adapter_path: str = None):
        """Merge LoRA adapters with base model and save"""
        
        if adapter_path is None:
            adapter_path = self.output_dir
        
        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("Loading LoRA adapters...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        
        print("Merging...")
        model = model.merge_and_unload()
        
        output_path = f"{self.output_dir}_merged"
        print(f"Saving merged model to {output_path}")
        model.save_pretrained(output_path)
        
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        tokenizer.save_pretrained(output_path)
        
        return output_path

if __name__ == "__main__":
    # Example usage with MLflow tracking
    from litigpt.training.tracking import MLflowTracker
    import yaml
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize MLflow
    tracker = MLflowTracker(
        experiment_name="reddit-chatbot-training",
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    )
    
    # Start run
    run_name = f"train_{config['data']['target_username']}"
    tracker.start_run(run_name=run_name, tags={
        'model': config['model']['base_model'],
        'user': config['data']['target_username']
    })
    
    # Log config
    tracker.log_config(config)
    
    # Initialize trainer
    trainer = RedditModelTrainer(
        model_name=config['model']['base_model'],
        output_dir=config['model']['output_dir']
    )
    
    # Train with MLflow logging
    trainer.train(
        data_dir=config['data']['training_dir'],
        num_epochs=config['training']['num_epochs'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate']
    )
    
    # Log model
    tracker.log_model(config['model']['output_dir'], model_name="reddit_bot")
    
    # End run
    tracker.end_run()
    
    print(f"\nView results at: {tracker.tracking_uri}")
    
    # Optional: merge LoRA adapters into the base model weights to produce a
    # single standalone model (no adapter files). Useful for Ollama/vLLM
    # deployments that don't support PEFT adapters natively. Costs extra VRAM
    # and disk space; skip unless you need a self-contained model file.
    # trainer.merge_and_save_full_model()