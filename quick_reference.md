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
```

## 🚀 Quick Start

```bash
# Full pipeline (local)
python run_pipeline.py --step all

# Step-by-step
python run_pipeline.py --step extract
python run_pipeline.py --step preprocess
python run_pipeline.py --step train
python run_pipeline.py --step eval
python run_pipeline.py --step deploy

# With Docker
docker-compose --profile training run --rm training
docker-compose --profile bot up -d bot
```

## 📊 MLflow Commands

```bash
# Start server
mlflow ui --port 5000

# With Docker
docker-compose up -d mlflow
# Access: http://localhost:5000

# Compare runs (Python)
from module_8_mlflow_tracking import MLflowTracker
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
python module_3_training.py
```

### AWS ECS (Bot)
```python
from module_12_cloud_deployment import AWSDeployer
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
# Interactive mode
from module_4_inference import RedditBotInference

bot = RedditBotInference(
    model_path="models/reddit_bot_lora",
    base_model="meta-llama/Llama-3.1-8B-Instruct"
)
bot.interactive_mode()

# Generate single response
response = bot.generate_response(
    context="user: What's your favorite game?",
    temperature=0.8
)
print(response)
```

## 📈 Model Evaluation

```python
from module_8_mlflow_tracking import ModelEvaluator

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
├── module_*.py          # Pipeline modules
├── Dockerfile.*         # Container definitions
├── docker-compose.yml   # Orchestration
├── config.yaml          # Configuration
├── .env                 # Secrets (don't commit!)
└── requirements.txt     # Dependencies
```

## 🔑 Key Files

| File | Purpose |
|------|---------|
| `module_1_data_extraction.py` | Extract user comments |
| `module_2_preprocessing.py` | Clean and format data |
| `module_3_training.py` | Fine-tune model |
| `module_4_inference.py` | Generate responses |
| `module_5_deployment.py` | Deploy bot to Reddit |
| `module_8_mlflow_tracking.py` | Track experiments |
| `run_pipeline.py` | Run full pipeline |
| `Dockerfile.training` | GPU training container |
| `Dockerfile.bot` | Bot deployment container |

## 💡 Tips

1. **Start small**: Use 500-1000 comments for initial testing
2. **Monitor MLflow**: Track metrics to improve quality
3. **Use Docker**: Ensures consistent environment
4. **Test locally**: Validate before cloud deployment
5. **Set rate limits**: Avoid Reddit API bans
6. **Add disclaimer**: Make it clear it's a bot
7. **Monitor costs**: Cloud resources add up
8. **Backup models**: Save to S3/GCS regularly

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
