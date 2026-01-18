# Reddit Chatbot - Complete Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│ Reddit Data  │────▶│   Module 1    │────▶│ Processed    │
│ (JSONL)      │     │ Extract Users │     │ User Data    │
│              │     │ - Single      │     │              │
│ comments.jsonl│     │ - Multiple    │     │ user_*.jsonl │
│ posts.jsonl  │     └───────────────┘     │ metadata.json│
└──────────────┘                           └──────┬───────┘
                                                  │
                                                  ▼
                                           ┌──────────────┐
                                           │   Module 2   │
                                           │ Preprocess   │
                                           │ - Clean      │
                                           │ - Context    │
                                           │ - Format     │
                                           └──────┬───────┘
                                                  │
                                                  ▼
                                           ┌──────────────┐
                                           │ Training     │
                                           │ Data         │
                                           │ train.jsonl  │
                                           │ val.jsonl    │
                                           └──────┬───────┘
                                                  │
                                                  ▼
┌─────────────┐                            ┌──────────────┐
│  MLflow     │◀───────────────────────────│   Module 3   │
│  Tracking   │    Logs metrics,           │   Training   │
│             │    parameters              │   QLoRA      │
│  Port 5000  │                            │   Fine-tune  │
└─────────────┘                            └──────┬───────┘
                                                  │
                                                  ▼
                                           ┌──────────────┐
                                           │ Trained      │
                                           │ Model        │
                                           │ *.safetensors│
                                           └──────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                    USER CLASSIFICATION                           │
│                    (Multi-User Only)                             │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐                           ┌──────────────┐
│ Processed    │────────────────────────▶ │  Module 13   │
│ User Data    │                           │  Classifier  │
│              │                           │  Builder     │
│ user_*.jsonl │                           │              │
└──────────────┘                           └──────┬───────┘
                                                  │
                                                  ▼
                                           ┌──────────────┐
                                           │ User         │
                                           │ Classifier   │
                                           │ Profiles     │
                                           │ (.pkl)       │
                                           └──────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE PIPELINE                           │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐
│ New Reddit   │
│ Comment      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Extract      │
│ Context      │
│ (3-5 parents)│
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌─────────────────────────────────┐
│ Multi-User?  │─Yes─▶│ Module 13: User Classifier     │
│              │     │                                 │
└──────┬───────┘     │ ┌─────────────────────────┐   │
       │ No          │ │ TF-IDF Similarity       │   │
       │             │ │ - Vectorize context     │   │
       │             │ │ - Compare to profiles   │   │
       │             │ │ - Rank users           │   │
       │             │ └───────────┬─────────────┘   │
       │             │             ▼                 │
       │             │ ┌─────────────────────────┐   │
       │             │ │ Keyword Matching        │   │
       │             │ │ - Check topic words     │   │
       │             │ │ - Match to users        │   │
       │             │ └───────────┬─────────────┘   │
       │             │             ▼                 │
       │             │ ┌─────────────────────────┐   │
       │             │ │ Hybrid Selection        │   │
       │             │ │ - Combine methods       │   │
       │             │ │ - Apply threshold       │   │
       │             │ └───────────┬─────────────┘   │
       │             └─────────────┼─────────────────┘
       │                           ▼
       │                    ┌──────────────┐
       │                    │ Selected User│
       │                    │ (username)   │
       │                    └──────┬───────┘
       │                           │
       └───────────────────────────┘
                                   │
                                   ▼
                            ┌──────────────┐
                            │ Build Prompt │
                            │              │
                            │ System: You  │
                            │ are {user}... │
                            │              │
                            │ User: {ctx}  │
                            └──────┬───────┘
                                   │
                                   ▼
                            ┌──────────────┐
                            │  Module 4    │
                            │  Inference   │
                            │              │
                            │  - Tokenize  │
                            │  - Generate  │
                            │  - Decode    │
                            └──────┬───────┘
                                   │
                                   ▼
                            ┌──────────────┐
                            │  Response    │
                            │  (in user's  │
                            │   style)     │
                            └──────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────┘

                            ┌──────────────┐
                            │   Reddit     │
                            │   Subreddit  │
                            └──────┬───────┘
                                   │ Stream comments
                                   ▼
                            ┌──────────────┐
                            │  Module 5    │
                            │  Bot Monitor │
                            │              │
                            │ - Filter     │
                            │ - Rate limit │
                            │ - Triggers   │
                            └──────┬───────┘
                                   │
                                   ├─────────────────────┐
                                   │                     │
                                   ▼                     ▼
                            ┌──────────────┐     ┌──────────────┐
                            │ Should       │     │  Context     │
                            │ Respond?     │     │  Extraction  │
                            │              │     │              │
                            │ - Score      │     │ - Get parents│
                            │ - Keywords   │     │ - Build thread│
                            │ - Probability│     └──────┬───────┘
                            └──────┬───────┘            │
                                   │ Yes               │
                                   └───────┬───────────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │  Inference   │
                                    │  Pipeline    │
                                    │  (see above) │
                                    └──────┬───────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │  Post Reply  │
                                    │              │
                                    │ + Disclaimer │
                                    │ + Logging    │
                                    └──────┬───────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │   Reddit     │
                                    │   Comment    │
                                    └──────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                   DOCKER ARCHITECTURE                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Docker Host                                   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │               Docker Network (reddit-bot-network)       │    │
│  │                                                          │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │    │
│  │  │   MLflow     │  │   Training   │  │     Bot      │ │    │
│  │  │   Server     │  │  Container   │  │  Container   │ │    │
│  │  │              │  │              │  │              │ │    │
│  │  │  Port 5000   │  │  GPU: Yes    │  │  GPU: No     │ │    │
│  │  │  Storage:    │  │  Runtime:    │  │  Runtime:    │ │    │
│  │  │  - mlruns/   │  │  nvidia      │  │  default     │ │    │
│  │  │  - artifacts/│  │              │  │              │ │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘ │    │
│  │                                                          │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                    Volumes                              │    │
│  │                                                          │    │
│  │  ./data        ←→  /workspace/data (Training)          │    │
│  │  ./models      ←→  /workspace/models (Training)        │    │
│  │  ./models      ←→  /app/models (Bot)                   │    │
│  │  ./mlruns      ←→  /mlflow/mlruns (MLflow)            │    │
│  │  ./logs        ←→  /app/logs (Bot)                     │    │
│  │  ./.env        ←→  /app/.env (Bot)                     │    │
│  │  ./config.yaml ←→  /workspace/config.yaml (Training)  │    │
│  │                                                          │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                   CLOUD DEPLOYMENT OPTIONS                       │
└─────────────────────────────────────────────────────────────────┘

Option 1: RunPod (Training)
┌──────────────────────────┐
│  RunPod GPU Instance     │
│  - RTX 4090 (24GB)       │
│  - Custom Docker Image   │
│  - Network Volume        │
│  - SSH Access            │
│                          │
│  Cost: $0.50-2.00/hr     │
└──────────────────────────┘

Option 2: AWS ECS (Bot)
┌──────────────────────────┐
│  AWS ECS Fargate         │
│  - CPU: 512-1024         │
│  - Memory: 1-2GB         │
│  - Auto-scaling          │
│  - CloudWatch logs       │
│                          │
│  Cost: $15-30/month      │
└──────────────────────────┘

Option 3: Google Cloud Run (Bot)
┌──────────────────────────┐
│  Cloud Run Service       │
│  - Serverless            │
│  - Auto-scale to 0       │
│  - Pay per use           │
│  - Secret Manager        │
│                          │
│  Cost: $5-15/month       │
└──────────────────────────┘

Option 4: Kubernetes (Multi-Bot)
┌──────────────────────────┐
│  Kubernetes Cluster      │
│  - Multiple bots         │
│  - Load balancing        │
│  - Rolling updates       │
│  - Persistent volumes    │
│                          │
│  Cost: $20-50/month      │
└──────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                      DATA FLOW                                   │
└─────────────────────────────────────────────────────────────────┘

Training Data Flow:
─────────────────
JSONL Files → Extract Users → Filter/Clean → Build Context → 
Format (ChatML) → Train (QLoRA) → Save Model → Log to MLflow

Multi-User Training:
───────────────────
JSONL Files → Extract Multiple Users → Create User-Tagged Pairs →
Format with Username in System Prompt → Train → Build Classifier

Inference Data Flow:
──────────────────
Reddit Comment → Extract Context → [Multi-User: Classify User] →
Build Prompt → Tokenize → Generate → Decode → Post Reply

User Classification:
──────────────────
Context → Vectorize (TF-IDF) → Compare to User Profiles →
Rank by Similarity → Select Top User → Return Username


┌─────────────────────────────────────────────────────────────────┐
│                    MODULE DEPENDENCIES                           │
└─────────────────────────────────────────────────────────────────┘

Module 1 (Extract)
     │
     ├──▶ Module 2 (Preprocess)
     │        │
     │        ├──▶ Module 3 (Training)
     │        │        │
     │        │        └──▶ Module 8 (MLflow)
     │        │
     │        └──▶ Module 13 (Classifier)
     │
     └──▶ Module 4 (Inference)
              │
              ├──▶ Module 13 (Classifier)
              │
              └──▶ Module 5 (Deployment)
                       │
                       └──▶ Module 4 (Inference)

Module 6 (Config) ─────▶ All Modules
Module 7 (Pipeline) ───▶ Orchestrates All


┌─────────────────────────────────────────────────────────────────┐
│                    PERFORMANCE METRICS                           │
└─────────────────────────────────────────────────────────────────┘

Training (RTX 4090, 2000 comments):
- Data extraction: ~5 min
- Preprocessing: ~10 min  
- Training (3 epochs): ~2 hours
- VRAM usage: ~12GB
- Total pipeline: ~2.5 hours

Inference (RTX 4090):
- Model loading: ~5 sec
- Response generation: ~2-3 sec
- With user classification: +10ms
- Throughput: 20-30 tokens/sec

Bot Deployment:
- Memory usage: ~4GB RAM
- CPU usage: <10% idle, ~40% generating
- Network: <1MB/hour
- Storage: ~10GB (model + cache)

Multi-User Overhead:
- Training time: +10% (more data)
- Model size: Same
- Inference time: +10ms (classification)
- Memory: +50MB (classifier profiles)
