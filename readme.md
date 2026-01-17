# Reddit Chatbot Pipeline - Complete Guide

A modular pipeline for creating a Reddit chatbot that mimics a specific user's writing style using fine-tuned language models.

## 📋 Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Breakdown](#module-breakdown)
- [Configuration](#configuration)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

---

## Overview

This pipeline takes Reddit conversation data (JSONL format) and creates a chatbot that can:
- Learn a specific user's writing style and tone
- Respond contextually to Reddit comments
- Deploy as an automated Reddit bot

**Pipeline Flow:**
```
Reddit JSONL Data → Extract User Data → Preprocess & Format → 
Fine-tune Model → Evaluate → Deploy to Reddit
```

---

## Requirements

### Hardware
- **Minimum**: 16GB RAM, 8GB VRAM GPU (RTX 3060, RTX 4060)
- **Recommended**: 32GB RAM, 16GB+ VRAM GPU (RTX 4090, A4000)
- **CPU Only**: Possible but very slow for training

### Software
- Python 3.10 or higher
- CUDA 11.8+ (for GPU acceleration)
- 50GB+ free disk space

---

## Installation

### 1. Clone and Setup

```bash
# Create project directory
mkdir reddit-chatbot
cd reddit-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install transformers datasets accelerate peft trl bitsandbytes
pip install pandas numpy jsonlines scikit-learn
pip install praw python-dotenv tqdm

# Install MLflow for experiment tracking
pip install mlflow

# Optional: for faster inference
pip install vllm

# Optional: for training monitoring
pip install tensorboard wandb
```

### 2b. Docker Setup (Alternative)

```bash
# Using Docker Compose for full stack
docker-compose up -d mlflow  # Start MLflow tracking server

# For training (with GPU)
docker-compose --profile training run --rm training

# For bot deployment
docker-compose --profile bot up -d bot
```

### 3. Project Structure

```
reddit-chatbot/
├── data/
│   ├── raw/              # Place your Reddit JSONL files here
│   ├── processed/        # Extracted user data
│   └── training/         # Formatted training data
├── models/               # Trained models
├── logs/                 # Bot logs
├── module_1_data_extraction.py
├── module_2_preprocessing.py
├── module_3_training.py
├── module_4_inference.py
├── module_5_deployment.py
├── module_6_config_setup.py
├── module_8_mlflow_tracking.py
├── run_pipeline.py
├── Dockerfile.training        # GPU training container
├── Dockerfile.bot            # Lightweight bot container
├── docker-compose.yml        # Orchestration
├── config.yaml
├── .env
└── requirements.txt
```

### 4. Configure Reddit API

Create `.env` file:

```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=MyBot/1.0 by /u/yourusername
REDDIT_USERNAME=your_bot_username
REDDIT_PASSWORD=your_bot_password
```

Get credentials at: https://www.reddit.com/prefs/apps

---

## Quick Start

### Option 1: Run Full Pipeline

```bash
# 1. Place your Reddit data in data/raw/
#    - comments.jsonl
#    - submissions.jsonl

# 2. Edit config.yaml with target username

# 3. Run complete pipeline
python run_pipeline.py --step all
```

### Option 2: Step-by-Step

```bash
# Step 1: Extract user data
python run_pipeline.py --step extract

# Step 2: Preprocess data
python run_pipeline.py --step preprocess

# Step 3: Train model (2-8 hours depending on data size)
python run_pipeline.py --step train

# Step 4: Evaluate model
python run_pipeline.py --step eval

# Step 5: Deploy bot
python run_pipeline.py --step deploy
```

---

## Module Breakdown

### Module 1: Data Extraction
**File**: `module_1_data_extraction.py`

Extracts target user's comments and posts from subreddit JSONL files.

```python
from module_1_data_extraction import RedditDataExtractor

extractor = RedditDataExtractor("data/raw")
user_data = extractor.extract_user_data("target_username")
extractor.save_processed_data(user_data, "data/processed/user_data.jsonl")
```

### Module 2: Preprocessing
**File**: `module_2_preprocessing.py`

Cleans data, builds conversation context, formats for training.

```python
from module_2_preprocessing import RedditDataPreprocessor

preprocessor = RedditDataPreprocessor()
user_data = preprocessor.filter_quality(user_data)
pairs = preprocessor.create_training_pairs(user_data, all_comments)
formatted = preprocessor.format_for_training(pairs, format_type="chatml")
```

### Module 3: Training
**File**: `module_3_training.py`

Fine-tunes language model using QLoRA for efficiency.

```python
from module_3_training import RedditModelTrainer

trainer = RedditModelTrainer(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    output_dir="models/reddit_bot_lora"
)
trainer.train(data_dir="data/training", num_epochs=3)
```

**Training Parameters:**
- **r=16**: LoRA rank (higher = more parameters, better quality, slower)
- **lora_alpha=32**: LoRA scaling factor
- **batch_size=4**: Adjust based on VRAM (lower if OOM errors)
- **num_epochs=3**: More epochs for smaller datasets

### Module 4: Inference
**File**: `module_4_inference.py`

Generate responses using the fine-tuned model.

```python
from module_4_inference import RedditBotInference

bot = RedditBotInference(
    model_path="models/reddit_bot_lora",
    base_model="meta-llama/Llama-3.1-8B-Instruct"
)

response = bot.generate_response(
    context="user1: What's your favorite game?",
    temperature=0.8
)
```

**Inference Parameters:**
- **temperature** (0.1-1.5): Higher = more creative/random
- **top_p** (0.1-1.0): Nucleus sampling threshold
- **max_new_tokens**: Maximum response length

### Module 5: Deployment
**File**: `module_5_deployment.py`

Deploys bot to monitor and respond on Reddit.

```python
from module_5_deployment import RedditBot

bot = RedditBot(
    model_path="models/reddit_bot_lora",
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    subreddit_name="test",
    bot_username="my_bot",
    reply_probability=0.2
)

bot.run()
```

---

## Configuration

### config.yaml

```yaml
data:
  target_username: "specific_reddit_user"  # User to mimic
  min_comment_length: 10
  max_comment_length: 512

model:
  base_model: "meta-llama/Llama-3.1-8B-Instruct"
  # Alternatives:
  # - "mistralai/Mistral-7B-Instruct-v0.2"  # Similar performance
  # - "microsoft/phi-3-mini-4k-instruct"     # Smaller, faster

training:
  num_epochs: 3
  batch_size: 4              # Reduce if out of memory
  learning_rate: 2.0e-4
  max_seq_length: 512

bot:
  subreddit: "test"
  reply_probability: 0.2     # 20% chance to reply
  cooldown_seconds: 120      # Wait 2 min between replies
```

---

## Usage

### Training Tips

**Small Dataset (< 500 comments)**
```yaml
training:
  num_epochs: 5
  learning_rate: 3.0e-4
```

**Large Dataset (> 2000 comments)**
```yaml
training:
  num_epochs: 2
  learning_rate: 1.0e-4
```

**Memory Issues**
```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 8  # Effective batch size = 16
```

### Interactive Testing

```python
from module_4_inference import RedditBotInference

bot = RedditBotInference("models/reddit_bot_lora", "meta-llama/Llama-3.1-8B-Instruct")
bot.interactive_mode()
```

### Custom Deployment

```python
from module_5_deployment import RedditBot

bot = RedditBot(
    model_path="models/reddit_bot_lora",
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    subreddit_name="mysubreddit",
    bot_username="mybot",
    trigger_keywords=["help", "question"],  # Only respond to these
    reply_probability=0.3,
    min_score_threshold=2  # Only reply to upvoted comments
)

bot.run()
```

---

## Troubleshooting

### Out of Memory (OOM)

**Training:**
```python
# Reduce batch size
batch_size: 2
gradient_accumulation_steps: 8

# Use smaller model
base_model: "microsoft/phi-3-mini-4k-instruct"
```

**Inference:**
```python
# Use 8-bit quantization instead of 4-bit
load_in_4bit=False
load_in_8bit=True
```

### Poor Response Quality

1. **Check training data quality**
   - Need at least 500+ comment pairs
   - User should have consistent style

2. **Adjust hyperparameters**
   ```yaml
   training:
     num_epochs: 5
     learning_rate: 3.0e-4
   ```

3. **Tune inference parameters**
   ```python
   temperature=0.9  # More creative
   repetition_penalty=1.2  # Less repetitive
   ```

### Bot Not Responding

1. **Check Reddit credentials** in `.env`
2. **Verify subreddit permissions** (some ban bots)
3. **Check trigger keywords** match actual comments
4. **Increase reply_probability** for testing

### Slow Training

- **Use smaller max_seq_length**: 256 instead of 512
- **Use gradient checkpointing**: Already enabled
- **Try Unsloth** for 2x faster training:
  ```bash
  pip install unsloth
  ```

---

## Advanced Topics

### Experiment Tracking with MLflow

**View Training Runs:**
```bash
# Start MLflow UI
mlflow ui --port 5000

# Or with Docker
docker-compose up -d mlflow
# Access at http://localhost:5000
```

**Compare Experiments:**
```python
from module_8_mlflow_tracking import MLflowTracker

tracker = MLflowTracker()
best_runs = tracker.compare_runs(metric="val_loss", n_best=5)
```

**Load Best Model:**
```python
best_model = tracker.load_best_model(metric="val_loss")
```

### Docker Deployment

**Training Container (GPU):**
```bash
# Build
docker build -f Dockerfile.training -t reddit-training .

# Run training
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/models:/workspace/models \
  reddit-training
```

**Bot Container (CPU):**
```bash
# Build
docker build -f Dockerfile.bot -t reddit-bot .

# Run bot
docker run -d \
  --name reddit-bot \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/.env:/app/.env \
  --restart unless-stopped \
  reddit-bot
```

**Full Stack with Docker Compose:**
```bash
# Start MLflow + Bot
docker-compose --profile bot up -d

# Run training
docker-compose --profile training run --rm training

# View logs
docker-compose logs -f bot

# Stop all
docker-compose down
```

### Cloud Deployment

**AWS ECS:**
```python
from module_12_cloud_deployment import AWSDeployer

deployer = AWSDeployer(region="us-east-1")
repo_uri = deployer.create_ecr_repository()
# Build and push Docker image to ECR
# Then deploy to ECS
```

**Google Cloud Run:**
```bash
./deploy_gcp.sh
```

**Kubernetes:**
```bash
kubectl apply -f k8s-bot-deployment.yaml
kubectl logs -f -n reddit-bot deployment/reddit-bot
```

### Using Different Models

**Smaller/Faster (Phi-3):**
```yaml
model:
  base_model: "microsoft/phi-3-mini-4k-instruct"
```

**Larger/Better (Mixtral):**
```yaml
model:
  base_model: "mistralai/Mixtral-8x7B-Instruct-v0.1"
  # Requires 40GB+ VRAM or multiple GPUs
```

### Multi-GPU Training

```python
# Automatically uses all available GPUs
trainer.train()
```

### Faster Inference with vLLM

```python
from module_4_inference import RedditBotInferenceVLLM

bot = RedditBotInferenceVLLM("models/reddit_bot_lora_merged")
response = bot.generate_response(context, temperature=0.8)
```

### Merging LoRA Adapters

```python
from module_3_training import RedditModelTrainer

trainer = RedditModelTrainer("meta-llama/Llama-3.1-8B-Instruct", "models/output")
merged_path = trainer.merge_and_save_full_model("models/reddit_bot_lora")
```

### Custom Conversation Context

```python
from module_1_data_extraction import RedditDataExtractor

extractor = RedditDataExtractor("data/raw")
thread_data = extractor.build_conversation_threads(all_comments)
context = extractor.get_context_for_comment(comment, thread_data, max_context=5)
```

---

## Performance Benchmarks

**Training (RTX 4090, 2000 comments):**
- Preprocessing: 5 minutes
- Training (3 epochs): 2 hours
- VRAM Usage: ~12GB

**Inference (RTX 4090):**
- Response time: ~2-3 seconds
- Throughput: 20-30 tokens/second

**Training (RTX 3060, 1000 comments):**
- Training (3 epochs): 6 hours
- VRAM Usage: ~10GB

---

## Ethics & Best Practices

1. **Disclose bot identity** - Always make it clear responses are AI-generated
2. **Follow subreddit rules** - Check if bots are allowed
3. **Rate limiting** - Don't spam, use appropriate cooldowns
4. **Content filtering** - Implement safeguards against harmful content
5. **Privacy** - Don't impersonate real users deceptively
6. **Monitoring** - Regularly check bot responses for quality

---

## License & Credits

This pipeline uses:
- Transformers (Apache 2.0)
- PyTorch (BSD)
- PRAW (BSD)
- LoRA/QLoRA techniques

Ensure compliance with model licenses (e.g., Llama 3.1 requires Meta approval for commercial use).

---

## Support

For issues or questions:
1. Check troubleshooting section
2. Review module documentation
3. Check logs in `logs/reddit_bot.log`
4. Verify configuration in `config.yaml`

---

## Next Steps

1. **Improve quality**: Collect more training data
2. **Add features**: Context-aware responses, personality tuning
3. **Deploy at scale**: Use cloud infrastructure
4. **Monitor performance**: Track response quality over time