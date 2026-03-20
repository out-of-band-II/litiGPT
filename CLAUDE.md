- [Claude AI Project Context](#claude-ai-project-context)
  - [Project Overview](#project-overview)
  - [Project Architecture](#project-architecture)
    - [Core Pipeline](#core-pipeline)
    - [Infrastructure](#infrastructure)
  - [Key Technical Details](#key-technical-details)
    - [Model Training](#model-training)
    - [Multi-User System](#multi-user-system)
    - [Data Format](#data-format)
  - [File Structure](#file-structure)
  - [Configuration](#configuration)
  - [Common Tasks](#common-tasks)
    - [For Claude: Helping with Code Issues](#for-claude-helping-with-code-issues)
    - [For Claude: Adding Features](#for-claude-adding-features)
  - [Code Style \& Patterns](#code-style--patterns)
    - [Naming Conventions](#naming-conventions)
    - [Error Handling](#error-handling)
    - [Logging](#logging)
    - [Progress Indication](#progress-indication)
  - [Testing Strategy](#testing-strategy)
    - [Manual Testing Flow](#manual-testing-flow)
    - [Multi-User Testing](#multi-user-testing)
  - [Performance Benchmarks](#performance-benchmarks)
    - [Training (RTX 4090)](#training-rtx-4090)
    - [Inference](#inference)
    - [Memory Usage](#memory-usage)
  - [Important Design Decisions](#important-design-decisions)
    - [Why QLoRA?](#why-qlora)
    - [Why Multiple Users in One Model?](#why-multiple-users-in-one-model)
    - [Why TF-IDF + Keywords?](#why-tf-idf--keywords)
    - [Why Separate Training/Bot Containers?](#why-separate-trainingbot-containers)
  - [Environment Variables](#environment-variables)
  - [Documentation Files](#documentation-files)
    - [For Users](#for-users)
    - [For Developers](#for-developers)
    - [Templates](#templates)
  - [Dependencies](#dependencies)
    - [Core (Required)](#core-required)
    - [Optional](#optional)
    - [Cloud](#cloud)
  - [Known Limitations](#known-limitations)
  - [Future Enhancements (Not Yet Implemented)](#future-enhancements-not-yet-implemented)
  - [Getting Help as Claude](#getting-help-as-claude)
  - [Quick Commands Reference](#quick-commands-reference)
  - [Project Status](#project-status)
    - [Completed Features](#completed-features)
    - [Test Coverage](#test-coverage)
  - [License \& Ethics](#license--ethics)
  - [Contact \& Support](#contact--support)


# Claude AI Project Context

This file provides context for Claude (or other AI assistants) working on this Reddit Chatbot project.

## Project Overview

**Reddit Chatbot Pipeline** - A complete, production-ready system for creating Reddit bots that mimic specific users' writing styles using fine-tuned language models.

**Key Innovation:** One model can learn multiple personalities and automatically select which to use based on conversation context. The system is unified — prompts always include the username, whether training on one user or many.

## Project Architecture

### Core Pipeline

```
Data (JSONL) → Extract → Preprocess → Train → Evaluate → Deploy
                                        ↓
                                    MLflow Tracking
```

All source code lives in the `litigpt/` package, organized by function:

1. **Data** (`litigpt/data/`):
   - `extraction.py`: Extract user comments/posts from Reddit JSONL
   - `preprocessing.py`: Clean, build context, format for training
   - `preliminary.py`: Raw data conversion utilities (zstd → parquet)

2. **Training** (`litigpt/training/`):
   - `trainer.py`: Fine-tune with QLoRA (4-bit quantization + LoRA adapters)
   - `tracking.py`: MLflow experiment tracking (logs params, metrics, models)

3. **Inference** (`litigpt/inference/`):
   - `generator.py`: Generate responses (single-user and multi-user)
   - `classifier.py`: Multi-user personality selection (TF-IDF, Keywords, Hybrid)

4. **Deployment** (`litigpt/deployment/`):
   - `reddit_bot.py`: Deploy to Reddit (monitors subreddit, auto-selects personality)
   - `cloud.py`: Cloud deployment (AWS ECS, Google Cloud Run, Kubernetes)

5. **Interface** (`litigpt/interface/`):
   - `gradio_app.py`: Gradio chat UI
   - `ollama.py`: Ollama-compatible API server

6. **Config & Pipeline**:
   - `litigpt/config.py`: Pydantic configuration models (validated, typed)
   - `litigpt/pipeline.py`: Orchestrates entire workflow

### Infrastructure

- **Docker**: Separate containers for training (GPU) and deployment (CPU)
  - `Dockerfile.training`: CUDA + training dependencies
  - `Dockerfile.bot`: Lightweight inference container
  - `docker_compose.yaml`: Orchestrates MLflow + Training + Bot

- **Cloud GPU Training**: Google Colab & Kaggle templates
  - `colab_training.ipynb`: 13-cell complete pipeline
  - `kaggle_training.py`: All-in-one Kaggle script

## Key Technical Details

### Model Training
- **Technique**: QLoRA (4-bit quantization + LoRA adapters)
- **Base Models**: Llama 3.1 8B (default), Mistral 7B, Phi-3
- **LoRA Config**: r=16, alpha=32, targets all attention layers
- **Hardware**: Minimum 12GB VRAM (RTX 3060, T4)
- **Training Time**: 2-3 hours for 2000 comments on RTX 4090

### Multi-User System
**Training**: Each example includes username in system prompt (in Italian, since the bot targets an Italian subreddit)
```python
System: "Sei alice, un utente di Reddit. Rispondi nello stile e nel tono di scrittura di alice."
User: [context]
Assistant: [alice's response]
```

**Inference**: Classifier selects user, then generates with that user's prompt
```
Context → Classifier → "alice" → Generate as alice → Response
```

**Classification Methods:**
1. **TF-IDF**: Vectorizes context, compares to user profiles (cosine similarity)
2. **Keywords**: Matches topics to predefined user keywords
3. **Hybrid**: Keyword match if available, else TF-IDF

### Data Format
**Input**: Reddit JSONL exports
- `comments.jsonl`: All subreddit comments
- `submissions.jsonl`: All posts

**Training Format**: ChatML
```json
{
  "messages": [
    {"role": "system", "content": "You are alice..."},
    {"role": "user", "content": "Context with 3-5 parent comments"},
    {"role": "assistant", "content": "User's actual response"}
  ]
}
```

## File Structure

```
reddit-chatbot/
├── data/
│   ├── raw/                    # Input JSONL files
│   ├── processed/              # Extracted user data
│   └── training/               # Formatted for training
├── models/
│   ├── reddit_bot_lora/        # LoRA adapters
│   └── user_classifier.pkl     # Multi-user classifier
├── litigpt/                    # Main package
│   ├── data/                   # Extraction & preprocessing
│   ├── training/               # Fine-tuning & MLflow tracking
│   ├── inference/              # Generation & classification
│   ├── deployment/             # Reddit bot & cloud deployment
│   ├── interface/              # Gradio & Ollama chat UIs
│   ├── config.py               # Configuration setup
│   └── pipeline.py             # Pipeline orchestrator
├── launch_chat.py              # Quick-launch script
├── Dockerfile.training
├── Dockerfile.bot
├── docker_compose.yaml
├── config.yaml
└── .env                        # Secrets (not committed)
```

## Configuration

Configuration is validated via Pydantic models defined in `litigpt/config.py`.
Load with `Config.from_yaml("config.yaml")` — provides autocomplete, validation,
and typed attribute access (`config.data.target_usernames`).

There is no separate `multi_user` flag. The system is unified: prompts always
include the username. Single-user is just `target_usernames` with one entry.

```yaml
data:
  target_usernames:
    - "alice_tech"
    - "bob_gaming"
    - "charlie_fitness"

bot:
  available_users: ["alice_tech", "bob_gaming", "charlie_fitness"]
  user_classifier_path: "models/user_classifier.pkl"
```

## Common Tasks

### For Claude: Helping with Code Issues

**Module Dependencies:**
- `data.extraction` → `data.preprocessing` → `training.trainer`
- `inference.classifier` (standalone, uses processed data)
- `deployment.reddit_bot` uses `inference.generator` & `inference.classifier`

**When debugging:**
1. Verify data exists and is formatted correctly
2. Ensure GPU availability for training
3. Check system prompts include username
4. Validate config loads: `Config.from_yaml("config.yaml")`

**Common Issues:**
- OOM: Reduce batch_size, max_seq_length
- Poor quality: More training data (500+ comments)
- Wrong user selected: Retrain classifier, add keywords
- Bot not responding: Check Reddit credentials, rate limits

### For Claude: Adding Features

**To add a new user selection method:**
1. Create new class in `litigpt/inference/classifier.py`
2. Implement `predict_user(context) -> str` method
3. Update `HybridUserSelector` to include it
4. Add config option in `config.yaml`

**To add a new deployment target:**
1. Add deployment class in `litigpt/deployment/cloud.py`
2. Follow pattern: `create_*`, `deploy_*` methods
3. Add example in `deployment_guide.md`

**To support a new base model:**
1. Add to `config.yaml` model options
2. Test chat template compatibility
3. Adjust batch_size if different memory requirements
4. Update README with performance benchmarks

## Code Style & Patterns

### Naming Conventions
- **Classes**: PascalCase (`RedditDataExtractor`)
- **Functions**: snake_case (`extract_user_data`)
- **Packages**: snake_case organized by function (`litigpt/data/extraction.py`)
- **Config keys**: snake_case nested dicts

### Error Handling
```python
try:
    # Operation
except SpecificException as e:
    logging.error(f"Context: {e}")
    # Graceful degradation or re-raise
```

### Logging
```python
import logging
logging.info("✓ Success message")
logging.warning("⚠️ Warning message")
logging.error("❌ Error message")
```

### Progress Indication
```python
from tqdm import tqdm
for item in tqdm(items, desc="Processing"):
    # work
```

## Testing Strategy

### Manual Testing Flow
1. **Small dataset** (100 comments) → Full pipeline
2. **Validate** each module output
3. **Test inference** interactively
4. **Deploy locally** first
5. **Scale up** to full dataset

### Multi-User Testing
```python
# Test extraction
users_data = extractor.extract_multiple_users(['user1', 'user2'])

# Test classifier
classifier = build_user_classifier_from_data('data/processed')
results = classifier.classify_context("test context", top_k=3)

# Test inference
bot.interactive_mode(available_users=['user1', 'user2'])
```

## Performance Benchmarks

### Training (RTX 4090)
- 500 comments: 30-45 min
- 1000 comments: 1-1.5 hours
- 2000 comments: 2-3 hours
- 5000 comments: 5-7 hours

### Inference
- Model loading: ~5 seconds
- Response generation: 2-3 seconds
- With classifier: +10ms overhead
- Throughput: 20-30 tokens/second

### Memory Usage
- Training: 12GB VRAM minimum
- Inference: 6-8GB VRAM (4-bit) or 4GB RAM (CPU)
- Bot deployment: ~4GB RAM

## Important Design Decisions

### Why QLoRA?
- **Efficient**: 4-bit quantization reduces memory by 75%
- **Effective**: Minimal quality loss vs full fine-tuning
- **Accessible**: Works on consumer GPUs (RTX 3060+)

### Why Multiple Users in One Model?
- **Efficient**: Single model vs multiple models
- **Flexible**: Dynamic personality switching
- **Practical**: Easier deployment and maintenance

### Why TF-IDF + Keywords?
- **TF-IDF**: Captures nuanced style/vocabulary
- **Keywords**: Fast, interpretable, topic-based
- **Hybrid**: Best of both worlds

### Why Separate Training/Bot Containers?
- **Training**: Needs GPU, CUDA, large dependencies
- **Bot**: CPU-only, minimal size, production-ready
- **Separation**: Faster deploys, lower costs

## Environment Variables

Required in `.env`:
```bash
REDDIT_CLIENT_ID=...          # From reddit.com/prefs/apps
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=...
REDDIT_USERNAME=...           # Bot account
REDDIT_PASSWORD=...
MLFLOW_TRACKING_URI=...       # Optional: http://localhost:5000
```

## Documentation Files

### For Users
- `README.md`: Complete setup guide
- `QUICK_REFERENCE.md`: One-page cheat sheet
- `DEPLOYMENT_GUIDE.md`: Cloud deployment
- `MULTI_USER_GUIDE.md`: Multi-user setup
- `CLOUD_GPU_TRAINING.md`: Colab/Kaggle guide

### For Developers
- `ARCHITECTURE.md`: System diagrams
- `MULTI_USER_SUMMARY.md`: Multi-user tech details
- `CLAUDE.md`: This file

### Templates
- `colab_training.ipynb`: Google Colab notebook
- `kaggle_training.py`: Kaggle script
- `config.yaml`: Configuration template
- `.env.example`: Environment template

## Dependencies

### Core (Required)
- torch ≥2.0.0
- transformers ≥4.36.0
- datasets, accelerate, peft, trl, bitsandbytes
- polars, pyarrow, numpy, jsonlines, scikit-learn
- pydantic ≥2.0.0, pyyaml
- praw (Reddit API)

### Optional
- mlflow (experiment tracking)
- vllm (faster inference)
- unsloth (faster training)
- tensorboard, wandb (monitoring)

### Cloud
- boto3 (AWS)
- google-cloud-* (GCP)
- kubernetes (K8s)

## Known Limitations

1. **Training Data**: Needs 100+ comments per user minimum
2. **Context**: Limited to 3-5 parent comments (token limits)
3. **GPU**: Requires 12GB+ VRAM for training
4. **Reddit API**: Rate limited to 60 requests/minute
5. **Session Limits**: Colab (12h), Kaggle (9h)
6. **Model Size**: 10-15GB with adapters

## Future Enhancements (Not Yet Implemented)

- [ ] Real-time learning from new conversations
- [ ] Sentiment-aware user selection
- [ ] Multi-subreddit deployment
- [ ] Web UI for model testing
- [ ] Automatic hyperparameter tuning
- [ ] Voice/audio Reddit integration
- [ ] Cross-platform (Discord, Twitter)

## Getting Help as Claude

When asked about this project:

1. **Check context**: Single-user or multi-user mode?
2. **Reference correct module**: Map task to package (data/, training/, inference/, deployment/)
3. **Consider deployment**: Local, Docker, or cloud?
4. **Check docs**: Point to relevant .md file
5. **Provide examples**: Use code from existing modules
6. **Think modular**: Each module is independent
7. **Consider scale**: Colab for quick, local for production

## Quick Commands Reference

```bash
# Full pipeline
python -m litigpt.pipeline --step all

# Individual steps
python -m litigpt.pipeline --step extract
python -m litigpt.pipeline --step preprocess
python -m litigpt.pipeline --step train
python -m litigpt.pipeline --step deploy

# Run individual modules directly
python -m litigpt.training.trainer
python -m litigpt.inference.classifier
python -m litigpt.deployment.reddit_bot

# Chat interfaces
python launch_chat.py --model models/reddit_bot_lora
python launch_chat.py --interface ollama --model models/reddit_bot_lora

# Docker
docker-compose --profile training run --rm training
docker-compose --profile bot up -d bot

# MLflow
mlflow ui --port 5000
```

## Project Status

**Version**: 1.0 (January 2025)
**Status**: Production-ready
**Maintenance**: Active

### Completed Features
✅ Single-user training and deployment
✅ Multi-user training and automatic selection
✅ MLflow experiment tracking
✅ Docker containerization
✅ Cloud deployment scripts (AWS, GCP, K8s)
✅ Free GPU training (Colab, Kaggle)
✅ Comprehensive documentation
✅ Ready-to-use templates

### Test Coverage
- Manual testing: Comprehensive
- Unit tests: Not yet implemented
- Integration tests: Not yet implemented

## License & Ethics

**License**: Open source (specify in LICENSE file)

**Ethics**:
- Always disclose bot identity
- Don't impersonate users deceptively
- Follow subreddit rules
- Implement rate limiting
- Add content filters
- Respect privacy

## Contact & Support

For issues:
1. Check troubleshooting in README
2. Review relevant guide (DEPLOYMENT, MULTI_USER, etc.)
3. Check logs: `logs/reddit_bot.log`
4. Verify configuration: `config.yaml`
5. Test with small dataset first

---

**For Claude**: This project is well-structured with clear separation of concerns. Each module is self-contained. When helping users, focus on their specific use case (single vs multi-user, local vs cloud) and point them to the relevant documentation. The codebase is modular and extensible - encourage users to build on existing patterns rather than rewriting core functionality.

**Last Updated**: February 2026
