# Claude AI Project Context

This file provides context for Claude (or other AI assistants) working on this Reddit Chatbot project.

## Project Overview

**Reddit Chatbot Pipeline** - A complete, production-ready system for creating Reddit bots that mimic specific users' writing styles using fine-tuned language models.

**Key Innovation:** Supports both single-user and multi-user modes, where one model can learn multiple personalities and automatically select which to use based on conversation context.

## Project Architecture

### Core Pipeline (7 Main Modules)

```
Data (JSONL) → Extract → Preprocess → Train → Evaluate → Deploy
                                        ↓
                                    MLflow Tracking
```

1. **Module 1** (`module_1_data_extraction.py`): Extract user comments/posts from Reddit JSONL
   - Single user: `extract_user_data(username)`
   - Multi-user: `extract_multiple_users(usernames)`
   
2. **Module 2** (`module_2_preprocessing.py`): Clean, build context, format for training
   - Creates conversation pairs with 3-5 parent comments as context
   - Multi-user: Tags each pair with username
   
3. **Module 3** (`module_3_training.py`): Fine-tune with QLoRA (memory-efficient)
   - Uses 4-bit quantization + LoRA adapters
   - Default: Llama 3.1 8B Instruct
   
4. **Module 4** (`module_4_inference.py`): Generate responses
   - Single-user: `generate_response(context)`
   - Multi-user: `generate_as_user(context, username)`
   
5. **Module 5** (`module_5_deployment.py`): Deploy to Reddit
   - Monitors subreddit, filters comments, posts replies
   - Multi-user: Auto-selects personality via classifier
   
6. **Module 6** (`module_6_config_setup.py`): Configuration and environment
   
7. **Module 7** (`run_pipeline.py`): Orchestrates entire workflow

### Extended Modules

8. **Module 8** (`module_8_mlflow_tracking.py`): Experiment tracking
   - Logs parameters, metrics, models
   - Compare runs, load best model
   
9. **Module 13** (`module_13_user_classifier.py`): Multi-user personality selection ⭐
   - **TF-IDF Classifier**: Statistical similarity matching
   - **Keyword Selector**: Topic-based routing
   - **Hybrid**: Combines both methods

### Infrastructure

10. **Docker**: Separate containers for training (GPU) and deployment (CPU)
    - `Dockerfile.training`: CUDA + training dependencies
    - `Dockerfile.bot`: Lightweight inference container
    - `docker-compose.yml`: Orchestrates MLflow + Training + Bot
    
11. **Cloud Deployment** (`module_12_cloud_deployment.py`):
    - AWS ECS, Google Cloud Run, Kubernetes configs
    - RunPod training scripts
    
12. **Cloud GPU Training**: Google Colab & Kaggle templates
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
**Training**: Each example includes username in system prompt
```python
System: "You are alice, a Reddit user. Respond in alice's style."
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
│   │   ├── user1_data.jsonl
│   │   ├── user2_data.jsonl
│   │   └── users_metadata.json
│   └── training/               # Formatted for training
│       ├── train.jsonl
│       └── val.jsonl
├── models/
│   ├── reddit_bot_lora/        # LoRA adapters
│   └── user_classifier.pkl     # Multi-user classifier
├── module_*.py                 # Core modules
├── Dockerfile.*
├── docker-compose.yml
├── config.yaml
└── .env                        # Secrets (not committed)
```

## Configuration

### Single-User Mode
```yaml
data:
  target_username: "specific_user"
  multi_user: false

bot:
  multi_user: false
```

### Multi-User Mode
```yaml
data:
  multi_user: true
  target_usernames:
    - "alice_tech"
    - "bob_gaming"
    - "charlie_fitness"

bot:
  multi_user: true
  available_users: ["alice_tech", "bob_gaming", "charlie_fitness"]
  user_classifier_path: "models/user_classifier.pkl"
```

## Common Tasks

### For Claude: Helping with Code Issues

**Module Dependencies:**
- Module 1 → Module 2 → Module 3 (training)
- Module 1 → Module 13 (classifier)
- Module 4 uses Module 13 (multi-user)
- Module 5 uses Modules 4 & 13

**When debugging:**
1. Check which mode (single vs multi-user)
2. Verify data exists and is formatted correctly
3. Ensure GPU availability for training
4. Check system prompts include username (multi-user)

**Common Issues:**
- OOM: Reduce batch_size, max_seq_length
- Poor quality: More training data (500+ comments)
- Wrong user selected: Retrain classifier, add keywords
- Bot not responding: Check Reddit credentials, rate limits

### For Claude: Adding Features

**To add a new user selection method:**
1. Create new class in `module_13_user_classifier.py`
2. Implement `predict_user(context) -> str` method
3. Update `HybridUserSelector` to include it
4. Add config option in `config.yaml`

**To add a new deployment target:**
1. Add deployment class in `module_12_cloud_deployment.py`
2. Follow pattern: `create_*`, `deploy_*` methods
3. Add example in `DEPLOYMENT_GUIDE.md`

**To support a new base model:**
1. Add to `config.yaml` model options
2. Test chat template compatibility
3. Adjust batch_size if different memory requirements
4. Update README with performance benchmarks

## Code Style & Patterns

### Naming Conventions
- **Classes**: PascalCase (`RedditDataExtractor`)
- **Functions**: snake_case (`extract_user_data`)
- **Files**: snake_case with module number (`module_1_data_extraction.py`)
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
- pandas, numpy, jsonlines, scikit-learn
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
2. **Reference correct module**: Map task to module number
3. **Consider deployment**: Local, Docker, or cloud?
4. **Check docs**: Point to relevant .md file
5. **Provide examples**: Use code from existing modules
6. **Think modular**: Each module is independent
7. **Consider scale**: Colab for quick, local for production

## Quick Commands Reference

```bash
# Extract (single)
python module_1_data_extraction.py

# Extract (multi)
python -c "from module_1_data_extraction import *; ..."

# Train
python module_3_training.py

# Build classifier
python module_13_user_classifier.py

# Deploy
python module_5_deployment.py

# Full pipeline
python run_pipeline.py --step all

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

**Last Updated**: January 2025
