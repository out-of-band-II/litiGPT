# Reddit Chatbot - Quick Reference

One-page reference for common tasks and commands.

## 📦 Installation

```bash
# Basic setup
python -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Docker setup
docker-compose up -d mlflow

# Cloud GPU (Colab/Kaggle)
# See CLOUD_GPU_TRAINING.md for complete guide
# Templates: colab_training.ipynb, kaggle_training.py
```

## 🚀 Quick Start

```bash
# Full pipeline (local) - Single user
python -m litigpt.pipeline --step all

# Full pipeline - Multi-user
# 1. Edit config.yaml to enable multi_user and list target_usernames
# 2. Run pipeline
python -m litigpt.pipeline --step all

# Step-by-step
python -m litigpt.pipeline --step extract
python -m litigpt.pipeline --step preprocess
python -m litigpt.pipeline --step train
python -m litigpt.pipeline --step eval
python -m litigpt.pipeline --step deploy

# Build user classifier (multi-user only)
python -c "from litigpt.inference.classifier import build_user_classifier_from_data; \
classifier = build_user_classifier_from_data('data/processed'); \
classifier.save_profiles('models/user_classifier.pkl')"

# With Docker
docker-compose --profile training run --rm training
docker-compose --profile bot up -d bot
```

## 👥 Multi-User Commands

```bash
# Extract multiple users
python -c "from litigpt.data.extraction import RedditDataExtractor; \
extractor = RedditDataExtractor('data/raw'); \
users_data = extractor.extract_multiple_users(['alice', 'bob', 'charlie']); \
extractor.save_multi_user_data(users_data, 'data/processed')"

# Build and test classifier
python -m litigpt.inference.classifier

# Test user selection
python -c "from litigpt.inference.classifier import UserClassifier; \
classifier = UserClassifier(); \
classifier.load_profiles('models/user_classifier.pkl'); \
print(classifier.classify_context('your context here', top_k=3))"
```

## 📊 MLflow Commands

```bash
# Start server
mlflow ui --port 5000

# With Docker
docker-compose up -d mlflow
# Access: http://localhost:5000

# Compare runs (Python)
from litigpt.training.tracking import MLflowTracker
tracker = MLflowTracker()
tracker.compare_runs(metric="val_loss", n_best=5)
```

## 🐳 Docker Commands

```bash
# Build containers
docker build -f Dockerfile.training -t reddit-training .
docker build -f Dockerfile.bot -t reddit-bot .

# Training (GPU)
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/models:/workspace/models \
  reddit-training

# Bot (CPU)
docker run -d \
  --name reddit-bot \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/.env:/app/.env \
  --restart unless-stopped \
  reddit-bot

# View logs
docker logs -f reddit-bot

# Stop/restart
docker stop reddit-bot
docker restart reddit-bot

# Docker Compose
docker-compose up -d mlflow          # Start MLflow
docker-compose --profile bot up -d   # Start bot
docker-compose logs -f bot           # View logs
docker-compose down                  # Stop all
```

## ☁️ Cloud Deployment

### RunPod (Training)
```bash
# 1. Push image
docker push YOUR_DOCKERHUB/reddit-training:latest

# 2. Deploy on https://runpod.io
# 3. SSH and run:
python -m litigpt.training.trainer
```

### Google Colab (Free GPU Training)
```python
# Use colab_training.ipynb template
# 1. Upload to Google Colab
# 2. Upload your data files
# 3. Run all cells
# 4. Download trained model

# Or quick command:
!wget https://raw.githubusercontent.com/yourrepo/colab_training.ipynb
# Open in Colab and run
```

### Kaggle (Free GPU Training)
```python
# Use kaggle_training.py template
# 1. Create Kaggle dataset with your JSONL files
# 2. Create new notebook, enable GPU
# 3. Add your dataset
# 4. Paste kaggle_training.py
# 5. Run and commit to save output
```

### AWS ECS (Bot)
```python
from litigpt.deployment.cloud import AWSDeployer
deployer = AWSDeployer(region="us-east-1")
repo_uri = deployer.create_ecr_repository()
# Build, push, deploy...
```

### Google Cloud Run
```bash
gcloud run deploy reddit-bot \
  --image gcr.io/PROJECT/reddit-bot \
  --platform managed
```

### Kubernetes
```bash
kubectl apply -f k8s-bot-deployment.yaml
kubectl logs -f -n reddit-bot deployment/reddit-bot
kubectl scale -n reddit-bot deployment/reddit-bot --replicas=3
```

## 🔧 Configuration

### config.yaml
```yaml
data:
  target_username: "username_to_mimic"
  
model:
  base_model: "meta-llama/Llama-3.1-8B-Instruct"
  
training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 2.0e-4
  
bot:
  subreddit: "test"
  reply_probability: 0.2
  cooldown_seconds: 120
```

### .env
```env
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_USER_AGENT=MyBot/1.0
REDDIT_USERNAME=bot_username
REDDIT_PASSWORD=bot_password
MLFLOW_TRACKING_URI=http://localhost:5000
```

## 🧪 Testing

```python
# Interactive mode - Single user
from litigpt.inference.generator import RedditBotInference

bot = RedditBotInference(
    model_path="models/reddit_bot_lora",
    base_model="meta-llama/Llama-3.1-8B-Instruct"
)
bot.interactive_mode()

# Interactive mode - Multi-user
bot.interactive_mode(available_users=["alice", "bob", "charlie"])
# Usage: @alice what's your opinion on Python?

# Generate as specific user
response = bot.generate_as_user(
    context="What's your favorite game?",
    username="bob",
    temperature=0.8
)
print(response)

# Auto-select user based on context
from litigpt.inference.classifier import UserClassifier
classifier = UserClassifier()
classifier.load_profiles("models/user_classifier.pkl")

context = "What's the best Python library?"
user = classifier.predict_user(context)
response = bot.generate_as_user(context, username=user)
```

## 📈 Model Evaluation

```python
from litigpt.training.tracking import ModelEvaluator

evaluator = ModelEvaluator(bot_inference)

# Quality metrics
metrics = evaluator.evaluate_response_quality(test_data)
print(f"BLEU: {metrics['avg_bleu']:.3f}")
print(f"ROUGE-L: {metrics['avg_rougeL']:.3f}")

# Style similarity
style = evaluator.evaluate_style_consistency(test_data, user_samples)
print(f"Style similarity: {style['style_similarity']:.3f}")
```

## 🐛 Troubleshooting

### Out of Memory
```yaml
# config.yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 8
  max_seq_length: 256
```

### Slow Training
```bash
# Use smaller model
base_model: "microsoft/phi-3-mini-4k-instruct"

# Or install Unsloth
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
```

### Bot Not Responding
```python
# Check credentials
import praw
reddit = praw.Reddit(...)
print(reddit.user.me())

# Increase reply probability for testing
bot_config:
  reply_probability: 1.0
```

### Docker GPU Issues
```bash
# Check GPU access
docker run --gpus all nvidia/cuda:11.8.0-base nvidia-smi

# Install NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## 📁 File Structure

```
reddit-chatbot/
├── data/
│   ├── raw/              # Your JSONL files here
│   ├── processed/        # Extracted user data
│   └── training/         # Training data
├── models/               # Trained models
├── logs/                 # Bot logs
├── mlruns/              # MLflow experiments
├── litigpt/             # Package modules
├── Dockerfile.*         # Container definitions
├── docker-compose.yml   # Orchestration
├── config.yaml          # Configuration
├── .env                 # Secrets (don't commit!)
└── requirements.txt     # Dependencies
```

## 🔑 Key Files

| File | Purpose |
|------|---------|
| `litigpt/data/extraction.py` | Extract user comments (single/multi) |
| `litigpt/data/preprocessing.py` | Clean and format data |
| `litigpt/training/trainer.py` | Fine-tune model |
| `litigpt/inference/generator.py` | Generate responses (with user selection) |
| `litigpt/deployment/reddit_bot.py` | Deploy bot to Reddit |
| `litigpt/training/tracking.py` | Track experiments |
| `litigpt/inference/classifier.py` | Auto-select user personality |
| `litigpt/pipeline.py` | Run full pipeline |
| `Dockerfile.training` | GPU training container |
| `Dockerfile.bot` | Bot deployment container |

## 💡 Tips

1. **Start small**: Use 500-1000 comments for initial testing
2. **Multi-user**: 3-5 diverse users works best
3. **Monitor MLflow**: Track metrics to improve quality
4. **Use Docker**: Ensures consistent environment
5. **Test locally**: Validate before cloud deployment
6. **Set rate limits**: Avoid Reddit API bans
7. **Add disclaimer**: Make it clear it's a bot
8. **Monitor costs**: Cloud resources add up
9. **Backup models**: Save to S3/GCS regularly
10. **User diversity**: Choose users with different topics/styles

## 📊 Resource Requirements

| Task | RAM | VRAM | Time |
|------|-----|------|------|
| Data extraction | 8GB | - | 5 min |
| Training (1000 comments) | 16GB | 12GB | 2-4 hrs |
| Inference | 8GB | 6GB | 2-3 sec/response |
| Bot deployment | 4GB | - | Continuous |

## 🔗 Useful Links

- MLflow UI: http://localhost:5000
- Tensorboard: http://localhost:6006
- Reddit Apps: https://www.reddit.com/prefs/apps
- Hugging Face Models: https://huggingface.co/models

## 📞 Getting Help

1. Check `README.md` for detailed docs
2. Check `DEPLOYMENT_GUIDE.md` for cloud setup
3. Review logs: `logs/reddit_bot.log`
4. Check MLflow for training issues
5. Review Docker logs: `docker logs reddit-bot`

---

**Last Updated:** January 2026
