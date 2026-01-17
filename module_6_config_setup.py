"""
Module 6: Configuration and Setup
Installation requirements and environment setup
"""

# ============================================
# requirements.txt
# ============================================
"""
# Core ML libraries
torch>=2.0.0
transformers>=4.36.0
datasets>=2.14.0
accelerate>=0.24.0
peft>=0.7.0
trl>=0.7.0
bitsandbytes>=0.41.0

# Data processing
pandas>=2.0.0
numpy>=1.24.0
jsonlines>=4.0.0
scikit-learn>=1.3.0

# Reddit API
praw>=7.7.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.65.0

# Optional: faster inference
# vllm>=0.2.0

# Optional: training monitoring
# tensorboard>=2.14.0
# wandb>=0.15.0

# Optional: faster training
# unsloth @ git+https://github.com/unslothai/unsloth.git
"""

# ============================================
# .env template
# ============================================
"""
# Create a .env file with these variables:

# Reddit API Credentials
# Get from: https://www.reddit.com/prefs/apps
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=YourBotName/1.0 by /u/yourusername
REDDIT_USERNAME=your_bot_username
REDDIT_PASSWORD=your_bot_password

# Optional: Hugging Face token for gated models
# HF_TOKEN=your_huggingface_token
"""

# ============================================
# config.yaml - Training Configuration
# ============================================
"""
# Save as config.yaml

data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  training_dir: "data/training"
  target_username: "specific_reddit_user"
  min_comment_length: 10
  max_comment_length: 512
  min_score: 1

model:
  base_model: "meta-llama/Llama-3.1-8B-Instruct"
  # Alternatives:
  # - "mistralai/Mistral-7B-Instruct-v0.2"
  # - "microsoft/phi-3-mini-4k-instruct"
  output_dir: "models/reddit_bot_lora"

training:
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  max_seq_length: 512
  warmup_ratio: 0.05
  
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  
inference:
  max_new_tokens: 256
  temperature: 0.8
  top_p: 0.9
  top_k: 50
  repetition_penalty: 1.1

bot:
  subreddit: "test"
  trigger_keywords: []  # Empty list = respond to all
  reply_probability: 0.2
  min_score_threshold: 1
  cooldown_seconds: 120
  max_context_depth: 3
"""

# ============================================
# Directory structure setup
# ============================================

import os
from pathlib import Path

def setup_project_structure():
    """Create necessary directories"""
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/training",
        "models",
        "logs",
        "scripts",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")

# ============================================
# Installation script
# ============================================

def install_dependencies():
    """Install all required packages"""
    import subprocess
    import sys
    
    print("Installing dependencies...")
    
    # Install PyTorch with CUDA
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ])
    
    # Install other requirements
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ])
    
    print("Installation complete!")

# ============================================
# Quick setup script
# ============================================

def quick_setup():
    """Run initial setup"""
    
    print("=" * 50)
    print("Reddit Chatbot Setup")
    print("=" * 50)
    
    # Create directories
    print("\n1. Creating project structure...")
    setup_project_structure()
    
    # Check for .env
    print("\n2. Checking environment variables...")
    if not Path(".env").exists():
        print("⚠️  No .env file found!")
        print("Please create a .env file with your Reddit API credentials.")
        print("See the template in this file.")
    else:
        print("✓ .env file found")
    
    # Check CUDA
    print("\n3. Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  CUDA not available. Training will be slow on CPU.")
    except ImportError:
        print("⚠️  PyTorch not installed yet.")
    
    print("\n" + "=" * 50)
    print("Setup complete! Next steps:")
    print("1. Add your Reddit data to data/raw/")
    print("2. Configure your target username in config.yaml")
    print("3. Run: python run_pipeline.py")
    print("=" * 50)

if __name__ == "__main__":
    quick_setup()
