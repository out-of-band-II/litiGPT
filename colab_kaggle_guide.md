# Free Cloud GPU Training Guide

Train your Reddit chatbot using free GPUs from Google Colab or Kaggle.

## 📋 Table of Contents

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
| **Internet** | Yes | Yes | Yes |
| **Best For** | Quick training | Production | Datasets < 20GB |

**Recommendation:**
- **Colab Free**: Best for most users, 12 hours is enough for 2000-3000 comments
- **Kaggle**: Better GPU (P100), but 9 hour limit and no persistent storage
- **Colab Pro**: Only if you need longer sessions or bigger models

---

## Google Colab Setup

### Method 1: Quick Start (Notebook)

Create a new notebook and run these cells:

**Cell 1: Setup Environment**
```python
# Check GPU availability
!nvidia-smi

# Install dependencies
!pip install -q torch transformers datasets accelerate peft trl bitsandbytes
!pip install -q pandas numpy jsonlines scikit-learn tqdm pyyaml mlflow

# Clone your repository (if using git)
# !git clone https://github.com/yourusername/reddit-chatbot.git
# %cd reddit-chatbot

# Or upload files manually
from google.colab import files
print("Upload your modules and data...")
```

**Cell 2: Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')

# Create project directory in Drive
!mkdir -p "/content/drive/MyDrive/reddit_chatbot"
%cd "/content/drive/MyDrive/reddit_chatbot"

# Your data and models will persist here
!mkdir -p data/raw data/processed data/training models logs
```

**Cell 3: Upload Data**
```python
# Option 1: Upload directly
from google.colab import files
uploaded = files.upload()  # Upload comments.jsonl and submissions.jsonl
!mv *.jsonl data/raw/

# Option 2: Download from URL
!wget -O data/raw/comments.jsonl "YOUR_DATA_URL"
!wget -O data/raw/submissions.jsonl "YOUR_DATA_URL"

# Option 3: Copy from Drive
!cp "/content/drive/MyDrive/your_data/*.jsonl" data/raw/
```

**Cell 4: Create Config**
```python
config = """
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  training_dir: "data/training"
  target_username: "your_target_user"
  min_comment_length: 10
  max_comment_length: 512

model:
  base_model: "meta-llama/Llama-3.1-8B-Instruct"
  output_dir: "models/reddit_bot_lora"

training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 2.0e-4
  max_seq_length: 512
"""

with open('config.yaml', 'w') as f:
    f.write(config)
```

**Cell 5: Extract Data**
```python
%%writefile litigpt/data/extraction.py
# Paste entire Module 1 code here
# [Copy from your litigpt/data/extraction.py]

# Run extraction
!python -c "
from litigpt.data.extraction import RedditDataExtractor
extractor = RedditDataExtractor('data/raw')
user_data = extractor.extract_user_data('your_username')
extractor.save_processed_data(user_data, 'data/processed/user_data.jsonl')
print(f'Extracted {len(user_data)} items')
"
```

**Cell 6: Preprocess**
```python
%%writefile litigpt/data/preprocessing.py
# Paste entire Module 2 code here

# Run preprocessing
!python -c "
import pandas as pd
import jsonlines
from litigpt.data.preprocessing import RedditDataPreprocessor

preprocessor = RedditDataPreprocessor()
user_data = pd.read_json('data/processed/user_data.jsonl', lines=True)
user_data = preprocessor.filter_quality(user_data)

# Load all comments
with jsonlines.open('data/raw/comments.jsonl') as reader:
    all_comments = list(reader)

# Create pairs
pairs = preprocessor.create_training_pairs(user_data, all_comments)
formatted = preprocessor.format_for_training(pairs, format_type='chatml')
train, val = preprocessor.split_data(formatted)
preprocessor.save_training_data(train, val)

print(f'Training samples: {len(train)}, Validation: {len(val)}')
"
```

**Cell 7: Train Model**
```python
%%writefile litigpt/training/trainer.py
# Paste entire Module 3 code here

# Run training
!python -m litigpt.training.trainer
```

**Cell 8: Monitor Training**
```python
# Watch training logs in real-time
!tail -f logs/training.log

# Or check GPU usage
!watch -n 1 nvidia-smi
```

**Cell 9: Download Model**
```python
# After training, download to your computer
from google.colab import files
import shutil

# Zip the model
!zip -r reddit_bot_lora.zip models/reddit_bot_lora

# Download
files.download('reddit_bot_lora.zip')

# Or keep in Drive (already saved if you're in Drive folder)
print("Model saved in Google Drive: /content/drive/MyDrive/reddit_chatbot/models/")
```

### Method 2: All-in-One Script

Create `train_colab.py`:

```python
"""
Complete training script for Google Colab
Run this after uploading your data
"""

import os
import sys

# Setup
print("=== Setting up environment ===")
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/training", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Check GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Import modules (paste module code above this, or import from files)
from litigpt.data.extraction import RedditDataExtractor
from litigpt.data.preprocessing import RedditDataPreprocessor
from litigpt.training.trainer import RedditModelTrainer

# Configuration
TARGET_USER = "your_username"  # CHANGE THIS
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

print("\n=== Step 1: Data Extraction ===")
extractor = RedditDataExtractor("data/raw")
user_data = extractor.extract_user_data(TARGET_USER)
extractor.save_processed_data(user_data, "data/processed/user_data.jsonl")
print(f"Extracted {len(user_data)} items")

print("\n=== Step 2: Preprocessing ===")
import pandas as pd
import jsonlines

preprocessor = RedditDataPreprocessor()
user_data = pd.read_json("data/processed/user_data.jsonl", lines=True)
user_data = preprocessor.filter_quality(user_data)

with jsonlines.open("data/raw/comments.jsonl") as reader:
    all_comments = list(reader)

pairs = preprocessor.create_training_pairs(user_data, all_comments)
formatted = preprocessor.format_for_training(pairs, format_type="chatml")
train, val = preprocessor.split_data(formatted)
preprocessor.save_training_data(train, val)
print(f"Training: {len(train)}, Validation: {len(val)}")

print("\n=== Step 3: Training ===")
trainer = RedditModelTrainer(
    model_name=BASE_MODEL,
    output_dir="models/reddit_bot_lora"
)

trainer.train(
    data_dir="data/training",
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4
)

print("\n=== Training Complete! ===")
print("Model saved to: models/reddit_bot_lora")
```

Upload this script and run:
```python
!python train_colab.py
```

---

## Kaggle Setup

### Method 1: Kaggle Notebook

**Step 1: Create New Notebook**
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Enable GPU: Settings → Accelerator → GPU T4 x2 or P100

**Step 2: Upload Data as Dataset**
1. Create a dataset: https://www.kaggle.com/datasets
2. Upload your JSONL files
3. Make it private
4. In your notebook: Add Data → Your Dataset

**Step 3: Setup Code**

```python
# Cell 1: Install dependencies
!pip install -q transformers datasets accelerate peft trl bitsandbytes

# Cell 2: Setup paths
import os
os.chdir('/kaggle/working')

# Your data is in /kaggle/input/your-dataset-name/
DATA_DIR = "/kaggle/input/reddit-data"  # Adjust to your dataset name

# Create output dirs
!mkdir -p data/processed data/training models

# Copy data to working directory (Kaggle input is read-only)
!cp -r {DATA_DIR}/* data/raw/
```

**Step 4: Paste and Run Modules**

Same as Colab method - paste modules and run training.

**Step 5: Save Output**

```python
# Kaggle doesn't persist /kaggle/working
# Save to output (will be available after run)
import shutil
shutil.make_archive('/kaggle/working/reddit_bot_model', 'zip', 'models/reddit_bot_lora')

# Or commit notebook - outputs are saved automatically
```

### Method 2: Kaggle with Git

```python
# Clone your repo
!git clone https://github.com/yourusername/reddit-chatbot.git
%cd reddit-chatbot

# Setup
!mkdir -p data/raw
!cp /kaggle/input/reddit-data/*.jsonl data/raw/

# Install requirements
!pip install -r requirements.txt

# Run pipeline
!python -m litigpt.pipeline --step all
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
setInterval(ClickConnect, 60000)  // Every minute
```

Or install extension: "Colab Keep Alive"

**Kaggle:**
- No need - sessions stay alive if actively running

### 2. Optimize for Time Limits

**Reduce training time:**
```yaml
training:
  num_epochs: 2  # Instead of 3
  batch_size: 8  # Larger if GPU allows
  max_seq_length: 256  # Shorter sequences
```

**Use gradient accumulation:**
```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 8  # Effective batch size = 16
```

**Use smaller model:**
```yaml
model:
  base_model: "microsoft/phi-3-mini-4k-instruct"  # Faster than Llama
```

### 3. Checkpoint Saving

Save checkpoints to resume if session dies:

```python
# In training args
training_args = TrainingArguments(
    save_strategy="steps",
    save_steps=100,  # Save every 100 steps
    save_total_limit=2,  # Keep only 2 checkpoints
)

# Resume from checkpoint
trainer.train(resume_from_checkpoint="models/reddit_bot_lora/checkpoint-500")
```

### 4. Data Upload Optimization

**Large datasets:**
```python
# Use gdown for Google Drive links
!pip install gdown
!gdown --id YOUR_FILE_ID -O data/raw/comments.jsonl

# Or use wget for direct links
!wget -O data.zip "YOUR_URL"
!unzip data.zip -d data/raw/
```

**Kaggle datasets:**
- Pre-upload to Kaggle datasets (one-time, then reusable)
- Much faster than uploading each session

### 5. Monitor Training

```python
# Simple progress monitor
from tqdm.auto import tqdm
import time

def monitor_training():
    while True:
        if os.path.exists("models/reddit_bot_lora/trainer_state.json"):
            import json
            with open("models/reddit_bot_lora/trainer_state.json") as f:
                state = json.load(f)
                print(f"Epoch: {state.get('epoch', 0):.2f}")
                print(f"Loss: {state.get('log_history', [{}])[-1].get('loss', 'N/A')}")
        time.sleep(60)

# Run in background
import threading
thread = threading.Thread(target=monitor_training)
thread.daemon = True
thread.start()
```

### 6. Multi-User Training

```python
# Extract multiple users
users = ["user1", "user2", "user3"]
users_data = extractor.extract_multiple_users(users, min_comments_per_user=100)

# Build classifier
from litigpt.inference.classifier import build_user_classifier_from_data
classifier = build_user_classifier_from_data("data/processed")
classifier.save_profiles("models/user_classifier.pkl")

# Download both
!zip -r models.zip models/
from google.colab import files
files.download('models.zip')
```

### 7. Bandwidth Optimization

```python
# Use 4-bit model loading (smaller download)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map="auto"
)

# Cache models to avoid re-downloading
!mkdir -p /root/.cache/huggingface
# Models cached here persist across Colab sessions (sometimes)
```

---

## Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
batch_size: 2
gradient_accumulation_steps: 8

# Reduce sequence length
max_seq_length: 256

# Use gradient checkpointing (already enabled)
gradient_checkpointing: True

# Clear cache between runs
import torch
torch.cuda.empty_cache()
```

### Session Timeout

**Colab:**
- Training takes > 12 hours → Use Colab Pro or split into multiple sessions
- Save checkpoints frequently
- Keep browser tab active

**Kaggle:**
- 9 hour limit is hard → Optimize training time
- Save intermediate outputs
- Can create new notebook and resume

### Slow Downloads

```python
# Use mirror sites for Hugging Face
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Or pre-download models
from huggingface_hub import snapshot_download
snapshot_download("meta-llama/Llama-3.1-8B-Instruct", local_dir="./model_cache")
```

### Kaggle Dataset Issues

```python
# Kaggle input is read-only
# Copy to working directory first
!cp -r /kaggle/input/your-data/* data/raw/

# Or work directly with input paths
extractor = RedditDataExtractor("/kaggle/input/your-data")
```

---

## Complete Colab Notebook Template

```python
# ============================================
# REDDIT CHATBOT TRAINING - GOOGLE COLAB
# ============================================

# === CELL 1: Setup ===
!pip install -q torch transformers datasets accelerate peft trl bitsandbytes
!pip install -q pandas numpy jsonlines scikit-learn tqdm pyyaml

from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive
!mkdir -p reddit_chatbot
%cd reddit_chatbot

!nvidia-smi

# === CELL 2: Upload Data ===
from google.colab import files
!mkdir -p data/raw

print("Upload comments.jsonl:")
uploaded = files.upload()
!mv comments.jsonl data/raw/

print("Upload submissions.jsonl:")
uploaded = files.upload()
!mv submissions.jsonl data/raw/

# === CELL 3: Configuration ===
TARGET_USER = "your_username"  # CHANGE THIS

config = f"""
data:
  target_username: "{TARGET_USER}"
  min_comment_length: 10
  max_comment_length: 512

model:
  base_model: "meta-llama/Llama-3.1-8B-Instruct"
  output_dir: "models/reddit_bot_lora"

training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 2.0e-4
  max_seq_length: 512
"""

with open('config.yaml', 'w') as f:
    f.write(config)

# === CELL 4: Paste Modules ===
# Paste litigpt/data/extraction.py here
%%writefile litigpt/data/extraction.py
[PASTE MODULE 1 CODE]

%%writefile litigpt/data/preprocessing.py
[PASTE MODULE 2 CODE]

%%writefile litigpt/training/trainer.py
[PASTE MODULE 3 CODE]

# === CELL 5: Run Training ===
!python -c "
import yaml
from litigpt.data.extraction import RedditDataExtractor
from litigpt.data.preprocessing import RedditDataPreprocessor
from litigpt.training.trainer import RedditModelTrainer
import pandas as pd
import jsonlines

with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Extract
print('Extracting...')
extractor = RedditDataExtractor('data/raw')
user_data = extractor.extract_user_data(config['data']['target_username'])
extractor.save_processed_data(user_data, 'data/processed/user_data.jsonl')

# Preprocess
print('Preprocessing...')
preprocessor = RedditDataPreprocessor()
user_data = pd.read_json('data/processed/user_data.jsonl', lines=True)
user_data = preprocessor.filter_quality(user_data)

with jsonlines.open('data/raw/comments.jsonl') as reader:
    all_comments = list(reader)

pairs = preprocessor.create_training_pairs(user_data, all_comments)
formatted = preprocessor.format_for_training(pairs)
train, val = preprocessor.split_data(formatted)
preprocessor.save_training_data(train, val)

# Train
print('Training...')
trainer = RedditModelTrainer(
    model_name=config['model']['base_model'],
    output_dir=config['model']['output_dir']
)
trainer.train(data_dir='data/training', num_epochs=3)

print('Done!')
"

# === CELL 6: Download Model ===
!zip -r reddit_bot_lora.zip models/reddit_bot_lora
from google.colab import files
files.download('reddit_bot_lora.zip')

print("Training complete! Model downloaded.")
print(f"Also saved in Drive: {os.getcwd()}/models/")
```

---

## Comparison: Colab vs Kaggle vs Local

| Aspect | Colab Free | Kaggle | Local (RTX 4090) |
|--------|-----------|--------|------------------|
| Setup time | 5 min | 5 min | 30 min |
| Training (2000 comments) | 2-3 hours | 2-3 hours | 2 hours |
| Data upload | Slow | Fast (datasets) | N/A |
| Session limit | 12 hours | 9 hours | Unlimited |
| Cost | Free | Free | $1600 hardware |
| Persistence | Google Drive | No | Yes |
| Convenience | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

**Bottom line:** Use Colab for most training, Kaggle for datasets you already have uploaded, Local only if you need to train frequently.

---

## Next Steps

1. ✅ Start with Colab Free
2. ✅ Upload small test dataset (100-200 comments)
3. ✅ Run complete pipeline
4. ✅ Validate model works
5. ✅ Scale up to full dataset
6. ✅ Download and deploy locally

For more details, see the main README and deployment guides!
