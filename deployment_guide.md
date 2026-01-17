# Reddit Chatbot - Deployment Guide

Complete guide for deploying your Reddit chatbot locally and to the cloud.

## Table of Contents

1. [Local Deployment](#local-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [MLflow Experiment Tracking](#mlflow-experiment-tracking)
5. [Production Best Practices](#production-best-practices)

---

## Local Deployment

### Quick Start (Development)

```bash
# 1. Train model locally
python module_3_training.py

# 2. Test interactively
python module_4_inference.py

# 3. Run bot
python module_5_deployment.py
```

### With MLflow Tracking

```bash
# 1. Start MLflow server
mlflow ui --port 5000 &

# 2. Set tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# 3. Train with tracking
python module_3_training.py

# 4. View experiments at http://localhost:5000
```

---

## Docker Deployment

### Architecture

```
┌─────────────────┐
│  MLflow Server  │  Port 5000 - Experiment tracking UI
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼────┐ ┌──▼──────┐
│Training│ │   Bot   │
│ (GPU)  │ │  (CPU)  │
└────────┘ └─────────┘
```

### Setup

**1. Install Docker & NVIDIA Container Toolkit**

```bash
# Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# NVIDIA Container Toolkit (for GPU training)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**2. Build Containers**

```bash
# Build training container
docker build -f Dockerfile.training -t reddit-training:latest .

# Build bot container
docker build -f Dockerfile.bot -t reddit-bot:latest .
```

**3. Run with Docker Compose**

```bash
# Start MLflow server
docker-compose up -d mlflow

# Access MLflow UI at http://localhost:5000

# Run training (one-time)
docker-compose --profile training run --rm training

# Start bot (runs continuously)
docker-compose --profile bot up -d bot

# View bot logs
docker-compose logs -f bot

# Stop everything
docker-compose down
```

### Manual Docker Commands

**Training (GPU required):**
```bash
docker run --gpus all \
  --name reddit-training \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/mlruns:/workspace/mlruns \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  reddit-training:latest \
  python module_3_training.py
```

**Bot (CPU only):**
```bash
docker run -d \
  --name reddit-bot \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/.env:/app/.env \
  --restart unless-stopped \
  reddit-bot:latest
```

**Interactive Testing:**
```bash
docker run -it --rm \
  -v $(pwd)/models:/app/models \
  reddit-bot:latest \
  python module_4_inference.py
```

---

## Cloud Deployment

### Option 1: RunPod (Training)

**Best for:** GPU training with pay-per-hour pricing

**Setup:**

1. Push Docker image to registry:
```bash
docker tag reddit-training:latest YOUR_DOCKERHUB/reddit-training:latest
docker push YOUR_DOCKERHUB/reddit-training:latest
```

2. Deploy on RunPod:
- Go to https://www.runpod.io/console/pods
- Select GPU (RTX 4090 recommended)
- Choose "Custom Container"
- Image: `YOUR_DOCKERHUB/reddit-training:latest`
- Volume mounts:
  - `/workspace/data` → Network Volume
  - `/workspace/models` → Network Volume
- Environment variables from `.env`

3. SSH into pod and run:
```bash
python module_3_training.py
```

**Cost:** ~$0.50-2.00/hour depending on GPU

### Option 2: AWS ECS (Bot Deployment)

**Best for:** Production bot deployment with auto-scaling

**Setup:**

```python
from module_12_cloud_deployment import AWSDeployer

# Initialize
deployer = AWSDeployer(region="us-east-1")

# Create ECR repository
repo_uri = deployer.create_ecr_repository("reddit-chatbot-bot")

# Build and push image
# aws ecr get-login-password --region us-east-1 | docker login ...
# docker tag reddit-bot:latest REPO_URI:latest
# docker push REPO_URI:latest

# Create task definition
task_arn = deployer.create_task_definition(
    image_uri=f"{repo_uri}:latest",
    task_name="reddit-bot-task",
    cpu="512",
    memory="1024"
)

# Create service
deployer.create_service(
    cluster_name="reddit-bots",
    service_name="reddit-bot-service",
    task_definition=task_arn,
    subnet_ids=["subnet-xxx"],
    security_group_ids=["sg-xxx"],
    desired_count=1
)
```

**Cost:** ~$15-30/month for continuous deployment

### Option 3: Google Cloud Run (Bot)

**Best for:** Serverless deployment, pay only when responding

```bash
# Build and push
docker build -f Dockerfile.bot -t gcr.io/PROJECT_ID/reddit-bot .
docker push gcr.io/PROJECT_ID/reddit-bot

# Deploy
gcloud run deploy reddit-bot \
  --image gcr.io/PROJECT_ID/reddit-bot \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --timeout 3600 \
  --set-env-vars MLFLOW_TRACKING_URI=https://your-mlflow.com \
  --set-secrets REDDIT_CLIENT_ID=reddit-secrets:latest
```

**Cost:** ~$5-15/month depending on activity

### Option 4: Kubernetes

**Best for:** Multi-bot deployment, high availability

```bash
# Apply configuration
kubectl apply -f k8s-bot-deployment.yaml

# View status
kubectl get pods -n reddit-bot

# View logs
kubectl logs -f -n reddit-bot deployment/reddit-bot

# Scale up
kubectl scale -n reddit-bot deployment/reddit-bot --replicas=3

# Update image
kubectl set image deployment/reddit-bot \
  -n reddit-bot \
  bot=your-registry/reddit-bot:v2
```

---

## MLflow Experiment Tracking

### Local Setup

```bash
# Start server
mlflow ui --backend-store-uri sqlite:///mlflow.db \
         --default-artifact-root ./mlartifacts \
         --host 0.0.0.0 \
         --port 5000
```

### Docker Setup

```bash
docker-compose up -d mlflow
# Access at http://localhost:5000
```

### Cloud MLflow (AWS)

```bash
# S3 for artifacts
mlflow server \
  --backend-store-uri postgresql://user:pass@host:5432/mlflow \
  --default-artifact-root s3://my-mlflow-bucket \
  --host 0.0.0.0
```

### Using MLflow

**View Experiments:**
```python
from module_8_mlflow_tracking import MLflowTracker

tracker = MLflowTracker()

# Compare runs
best_runs = tracker.compare_runs(metric="val_loss", n_best=5)

# Load best model
best_model = tracker.load_best_model(metric="val_loss")
```

**Track Custom Metrics:**
```python
tracker.start_run(run_name="experiment_1")
tracker.log_config(config)
tracker.log_training_metrics(epoch=1, train_loss=0.5, val_loss=0.6)
tracker.log_evaluation_metrics({'bleu': 0.45, 'rouge': 0.52})
tracker.end_run()
```

---

## Production Best Practices

### 1. Health Checks

Add to bot code:
```python
import threading
from flask import Flask

app = Flask(__name__)

@app.route('/health')
def health():
    return {'status': 'healthy', 'bot_running': bot.is_running()}

# Run in separate thread
threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8080)).start()
```

### 2. Monitoring

**CloudWatch (AWS):**
```python
import boto3

cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_data(
    Namespace='RedditBot',
    MetricData=[{
        'MetricName': 'ResponsesGenerated',
        'Value': 1,
        'Unit': 'Count'
    }]
)
```

**Prometheus:**
```python
from prometheus_client import Counter, start_http_server

responses_counter = Counter('bot_responses_total', 'Total responses generated')

start_http_server(9090)
responses_counter.inc()
```

### 3. Logging

```python
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger('reddit_bot')
handler = RotatingFileHandler(
    'logs/bot.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
logger.addHandler(handler)
```

### 4. Rate Limiting

```python
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=60, period=60)  # 60 calls per minute
def post_reply(comment, text):
    comment.reply(text)
```

### 5. Error Recovery

```python
def run_with_retry():
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            bot.run()
        except Exception as e:
            logger.error(f"Bot crashed: {e}")
            retry_count += 1
            time.sleep(60 * retry_count)  # Exponential backoff
```

### 6. CI/CD Pipeline

**GitHub Actions (.github/workflows/deploy.yml):**
```yaml
name: Deploy Bot

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build Docker image
        run: docker build -f Dockerfile.bot -t reddit-bot .
      
      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push your-registry/reddit-bot:latest
      
      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster reddit-bots \
            --service reddit-bot-service \
            --force-new-deployment
```

### 7. Secrets Management

**AWS Secrets Manager:**
```python
import boto3
import json

def get_reddit_credentials():
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='reddit-bot-credentials')
    return json.loads(response['SecretString'])
```

**HashiCorp Vault:**
```python
import hvac

client = hvac.Client(url='http://vault:8200', token='your-token')
secrets = client.secrets.kv.v2.read_secret_version(path='reddit-bot')
```

### 8. Backup & Recovery

```bash
# Backup models
aws s3 sync ./models s3://my-bot-backups/models/$(date +%Y%m%d)

# Backup MLflow
aws s3 sync ./mlruns s3://my-bot-backups/mlruns/$(date +%Y%m%d)

# Backup database
pg_dump mlflow > mlflow_backup_$(date +%Y%m%d).sql
```

### 9. Performance Optimization

**Model Quantization:**
```python
# Use 8-bit instead of 4-bit for better quality
bot = RedditBotInference(
    model_path="models/reddit_bot_lora",
    load_in_4bit=False,
    load_in_8bit=True
)
```

**Response Caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def generate_response_cached(context_hash):
    return bot.generate_response(context)
```

**Batch Processing:**
```python
# Process multiple comments at once
contexts = [get_context(c) for c in comments[:10]]
responses = bot.generate_batch(contexts)
```

---

## Cost Comparison

| Platform | Training Cost | Bot Cost (Monthly) | Notes |
|----------|--------------|-------------------|-------|
| **Local** | Electricity | $0 | Requires GPU hardware |
| **RunPod** | $0.50-2/hr | N/A | Pay per training hour |
| **AWS ECS** | N/A | $15-30 | Always-on deployment |
| **Google Cloud Run** | N/A | $5-15 | Serverless, pay per use |
| **AWS Lambda** | N/A | $2-10 | Event-driven only |
| **Kubernetes (GKE)** | Variable | $20-50 | Best for multi-bot |

---

## Troubleshooting

**Docker Issues:**
```bash
# Check GPU access
docker run --gpus all nvidia/cuda:11.8.0-base nvidia-smi

# View container logs
docker logs reddit-bot

# Enter container
docker exec -it reddit-bot bash
```

**MLflow Connection:**
```bash
# Test connection
curl http://localhost:5000/health

# Check environment
echo $MLFLOW_TRACKING_URI
```

**Cloud Deployment:**
```bash
# AWS ECS logs
aws logs tail /ecs/reddit-bot-task --follow

# GCP Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision" --limit 50
```

---

## Next Steps

1. **Start with local deployment** to validate everything works
2. **Use Docker** for reproducible environments
3. **Deploy to cloud** for production use
4. **Monitor with MLflow** to track model performance
5. **Set up CI/CD** for automated deployments

For questions or issues, check the main README troubleshooting section.
