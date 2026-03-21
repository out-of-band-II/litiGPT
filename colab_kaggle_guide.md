# Free Cloud GPU Training Guide

Train your Reddit chatbot using free GPUs from Google Colab or Kaggle.

## Table of Contents

- [Platform Comparison](#platform-comparison)
- [Google Colab Setup](#google-colab-setup)
- [Kaggle Setup](#kaggle-setup)
- [Tips & Tricks](#tips--tricks)
- [Troubleshooting](#troubleshooting)

---

## Platform Comparison

| Feature | Google Colab Free | Colab Pro | Kaggle |
|---------|------------------|-----------|--------|
| **GPU** | T4 (16GB) | V100/A100 | P100/T4 (16GB) |
| **RAM** | 12GB | 32GB+ | 30GB |
| **Session** | ~12 hours | ~24 hours | 9 hours |
| **Storage** | 15GB (Drive) | 200GB | 20GB (temp) |
| **Cost** | Free | $10/month | Free |
| **Best For** | Quick training | Production | Datasets < 20GB |

**Recommendation:**
- **Colab Free**: Best for most users, 12 hours is enough for 2000-3000 comments
- **Kaggle**: Better GPU (P100), but 9 hour limit and no persistent storage
- **Colab Pro**: Only if you need longer sessions or bigger models

---

## Google Colab Setup

### Method 1: Pipeline (Recommended)

The simplest approach is to clone the repo and run the pipeline directly.

**Cell 1: Check GPU and install dependencies**
```python
!nvidia-smi

!pip install -q transformers datasets accelerate peft trl bitsandbytes
!pip install -q polars jsonlines scikit-learn tqdm pyyaml pydantic mlflow praw
```

**Cell 2: Mount Drive and clone repo**
```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive
!git clone https://github.com/yourusername/litiGPT.git
%cd litiGPT

!mkdir -p data/raw data/processed data/training models logs
```

**Cell 3: Upload data**
```python
# Option 1: Upload directly
from google.colab import files
print("Upload your .parquet data files:")
uploaded = files.upload()
!mv *.parquet data/raw/

# Option 2: Copy from Drive
# !cp "/content/drive/MyDrive/your_data/*.parquet" data/raw/
```

**Cell 4: Create config**
```python
config = """
data:
  raw_dir: "data/raw"
  submission_filename: "litigi_submissions.parquet"
  comments_filename: "litigi_comments.parquet"
  processed_dir: "data/processed"
  training_dir: "data/training"
  target_usernames:
    - "your_target_user"
  min_comment_length: 10
  max_comment_length: 512
  min_score: 1

model:
  base_model: "microsoft/phi-3-mini-4k-instruct"
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
  available_users: []
  trigger_keywords: []
  reply_probability: 0.2
  min_score_threshold: 1
  cooldown_seconds: 120
  max_context_depth: 3
"""

with open('config.yaml', 'w') as f:
    f.write(config)
print("Config written.")
```

**Cell 5: Run full pipeline**
```python
!python -m litigpt.pipeline --step extract
!python -m litigpt.pipeline --step preprocess
!python -m litigpt.pipeline --step train
```

Or all at once:
```python
!python -m litigpt.pipeline --step all
```

**Cell 6: Download model**
```python
!zip -r reddit_bot_lora.zip models/reddit_bot_lora
from google.colab import files
files.download('reddit_bot_lora.zip')
print("Model also saved in Drive:", f"{import os; os.getcwd()}/models/")
```

---

### Method 2: Step-by-Step (Manual API)

For debugging or running individual steps programmatically.

**Extract**
```python
import polars as pl
from litigpt.data.extraction import RedditDataExtractor

extractor = RedditDataExtractor("data/raw")
users_data = extractor.extract_multiple_users(
    usernames=["your_target_user"],
    comments_file="litigi_comments.parquet",
    posts_file="litigi_submissions.parquet"
)
extractor.save_multi_user_data(users_data, "data/processed")

for user, df in users_data.items():
    print(f"{user}: {len(df)} items")
```

**Preprocess**
```python
import polars as pl
from pathlib import Path
from litigpt.data.extraction import RedditDataExtractor
from litigpt.data.preprocessing import RedditDataPreprocessor

preprocessor = RedditDataPreprocessor(min_length=10, max_length=512)

# Load per-user parquet files saved by extraction
processed_dir = Path("data/processed")
users_data = {
    p.stem.replace("_data", ""): pl.read_parquet(p)
    for p in sorted(processed_dir.glob("*_data.parquet"))
}

# Load all comments for context building
extractor = RedditDataExtractor("data/raw")
all_comments = extractor.load_data("litigi_comments.parquet")

# Filter quality per user
users_data = {u: preprocessor.filter_quality(d) for u, d in users_data.items()}

# Create training pairs and format
pairs = preprocessor.create_multi_user_training_pairs(users_data, all_comments)
formatted = preprocessor.format_for_training(pairs, format_type="chatml")
train, val = preprocessor.split_data(formatted, train_ratio=0.9)
preprocessor.save_training_data(train, val, output_dir="data/training")

print(f"Training: {len(train)}, Validation: {len(val)}")
```

**Train**
```python
from litigpt.training.trainer import RedditModelTrainer

trainer = RedditModelTrainer(
    model_name="microsoft/phi-3-mini-4k-instruct",
    output_dir="models/reddit_bot_lora"
)

trainer.train(
    data_dir="data/training",
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    max_seq_length=512,
    report_to="none",  # or "mlflow" if you have MLflow running
)
```

---

## Kaggle Setup

### Method 1: Kaggle Notebook with Pipeline

**Step 1: Create New Notebook**
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Enable GPU: Settings -> Accelerator -> GPU T4 x2 or P100

**Step 2: Upload Data as Dataset**
1. Create a dataset: https://www.kaggle.com/datasets
2. Upload your `.parquet` files (`litigi_comments.parquet`, `litigi_submissions.parquet`)
3. Make it private
4. In your notebook: Add Data -> Your Dataset

**Step 3: Setup**

```python
# Cell 1: Install
!pip install -q transformers datasets accelerate peft trl bitsandbytes
!pip install -q polars jsonlines scikit-learn tqdm pyyaml pydantic mlflow praw

# Cell 2: Clone repo
import os
os.chdir('/kaggle/working')
!git clone https://github.com/yourusername/litiGPT.git
os.chdir('litiGPT')

# Copy data (Kaggle input is read-only)
DATA_DIR = "/kaggle/input/your-dataset-name"  # adjust to your dataset name
!mkdir -p data/raw
!cp {DATA_DIR}/*.parquet data/raw/

# Cell 3: Create config (same as Colab Cell 4 above, adjust username)

# Cell 4: Run pipeline
!python -m litigpt.pipeline --step extract
!python -m litigpt.pipeline --step preprocess
!python -m litigpt.pipeline --step train
```

**Step 4: Save Output**
```python
import shutil
shutil.make_archive('/kaggle/working/reddit_bot_model', 'zip', 'litiGPT/models/reddit_bot_lora')
# Output is saved automatically when you commit the notebook
```

---

## Tips & Tricks

### 1. Keep Sessions Alive

**Colab:**
```javascript
// Run in browser console (F12)
function ClickConnect(){
  console.log("Clicking connect...");
  document.querySelector("colab-connect-button").shadowRoot.querySelector("#connect").click()
}
setInterval(ClickConnect, 60000)
```

**Kaggle:** No need - sessions stay alive if actively running.

### 2. Optimize for Time Limits

```yaml
training:
  num_epochs: 2            # instead of 3
  batch_size: 2
  gradient_accumulation_steps: 8   # effective batch = 16
  max_seq_length: 256      # shorter sequences = faster

model:
  base_model: "microsoft/phi-3-mini-4k-instruct"  # faster than Llama 8B
```

### 3. Resume from Checkpoint

Training saves checkpoints automatically (`save_steps: 100`). To resume if the session dies:
```python
trainer.train(resume_from_checkpoint="models/reddit_bot_lora/checkpoint-500")
```

### 4. Multi-User Training

Just add more usernames to the config:
```yaml
data:
  target_usernames:
    - "user1"
    - "user2"
    - "user3"
```

The pipeline handles extraction, preprocessing, and training for all users automatically. A single model learns all personalities.

### 5. Monitor Training

```python
import json, time

def monitor():
    state_file = "models/reddit_bot_lora/trainer_state.json"
    while True:
        if os.path.exists(state_file):
            with open(state_file) as f:
                state = json.load(f)
            history = state.get('log_history', [])
            if history:
                last = history[-1]
                print(f"Epoch: {last.get('epoch', 0):.2f}  Loss: {last.get('loss', 'N/A')}")
        time.sleep(60)

import threading
threading.Thread(target=monitor, daemon=True).start()
```

### 6. Bandwidth / Cache

```python
# Use HF mirror if downloads are slow
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Disable symlink warning on Windows-based environments
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
```

---

## Troubleshooting

### Out of Memory (OOM)

Reduce in `config.yaml`:
```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 16
  max_seq_length: 256

# Then clear cache before retrying:
import torch; torch.cuda.empty_cache()
```

### Session Timeout

- **Colab**: Training > 12 hours -> use Colab Pro or split sessions. Checkpoints save every 100 steps.
- **Kaggle**: Hard 9-hour limit -> use `phi-3-mini` and reduce epochs.

### Slow Downloads

```python
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Or pre-download the model in a separate cell
from huggingface_hub import snapshot_download
snapshot_download("microsoft/phi-3-mini-4k-instruct", local_dir="./model_cache")
```

### Kaggle Dataset Issues

```python
# Kaggle /kaggle/input is read-only - always copy first
!cp -r /kaggle/input/your-data/*.parquet data/raw/
```

### No module named 'litigpt'

Make sure you're in the repo root directory when running:
```python
import os
os.chdir('/content/drive/MyDrive/litiGPT')  # Colab
# or
os.chdir('/kaggle/working/litiGPT')          # Kaggle
```

---

## Comparison: Colab vs Kaggle vs Local

| Aspect | Colab Free | Kaggle | Local (RTX 4090) |
|--------|-----------|--------|------------------|
| Setup time | 5 min | 5 min | 30 min |
| Training (2000 comments) | 2-3 hours | 2-3 hours | 2 hours |
| Data upload | Slow | Fast (datasets) | N/A |
| Session limit | 12 hours | 9 hours | Unlimited |
| Cost | Free | Free | ~$1600 hardware |
| Persistence | Google Drive | No | Yes |

**Bottom line:** Use Colab for most training, Kaggle for datasets already uploaded there, local only if training frequently.

---

## Next Steps

1. Start with Colab Free
2. Upload a small test dataset (100-200 comments) and run the full pipeline
3. Validate model works with `--step eval`
4. Scale up to full dataset
5. Download and deploy locally

For more details, see the main README and deployment guides.
