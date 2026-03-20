# Multi-User Reddit Bot Guide

Train a single model to imitate multiple users and automatically select which personality to use based on context.

## 📋 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Setup](#setup)
- [Training](#training)
- [User Selection Methods](#user-selection-methods)
- [Deployment](#deployment)
- [Examples](#examples)

---

## Overview

Multi-user mode allows your bot to:
- **Learn multiple writing styles** in a single model
- **Automatically select** which user to impersonate based on conversation context
- **Switch personalities** dynamically based on topic/style
- **More diverse responses** by combining different user traits

### Single vs Multi-User

**Single-User Mode:**
```
Context: "What's your favorite Python library?"
Bot: "I really like pandas for data analysis!"
```

**Multi-User Mode:**
```
Context: "What's your favorite Python library?"
Bot (as tech_user): "I really like pandas for data analysis!"

Context: "Anyone play Valorant?"
Bot (as gamer_user): "Yeah! I'm stuck in Diamond trying to hit Ascendant"
```

---

## How It Works

### 1. Training Phase

The model learns to respond as different users by including the username in the system prompt:

```
System: Sei alice, un utente di Reddit. Rispondi nello stile e nel tono di scrittura di alice.
User: What do you think about AI?
Assistant: [alice's response style]

System: Sei bob, un utente di Reddit. Rispondi nello stile e nel tono di scrittura di bob.
User: What do you think about AI?
Assistant: [bob's response style]
```

### 2. Inference Phase

When generating a response, the bot:
1. Analyzes the conversation context
2. Selects which user's style best matches (via classifier)
3. Sets the system prompt to that user
4. Generates response in that user's style

### 3. User Selection

Three methods available:

**TF-IDF Classifier** (default):
- Builds statistical profiles of each user's vocabulary
- Compares context to user profiles
- Selects most similar user

**Keyword Matcher**:
- Define topics/keywords for each user
- Match context keywords to users
- Fast and interpretable

**Hybrid**:
- Combines both methods
- Keyword match takes priority, TF-IDF as fallback

---

## Setup

### 1. Configure Multiple Users

Edit `config.yaml`:

```yaml
data:
  target_usernames:
    - "alice_tech"      # Tech enthusiast
    - "bob_gaming"      # Gaming expert
    - "charlie_fitness" # Fitness guru

bot:
  available_users:
    - "alice_tech"
    - "bob_gaming"
    - "charlie_fitness"
  user_classifier_path: "models/user_classifier.pkl"
```

> **Note:** There is no separate `multi_user` flag. The system is unified: prompts always include the username, and multiple users are supported simply by listing them in `target_usernames` and `available_users`.

### 2. Extract User Data

```python
from litigpt.data.extraction import RedditDataExtractor

extractor = RedditDataExtractor("data/raw")

# Extract multiple users
users_data = extractor.extract_multiple_users(
    usernames=["alice_tech", "bob_gaming", "charlie_fitness"],
    min_comments_per_user=100
)

# Save
extractor.save_multi_user_data(users_data, "data/processed")
```

Or use the pipeline:

```bash
python -m litigpt.pipeline --step extract
```

---

## Training

### 1. Preprocess Multi-User Data

```python
from litigpt.data.preprocessing import RedditDataPreprocessor
import pandas as pd
import jsonlines
import json

preprocessor = RedditDataPreprocessor()

# Load all users
with open("data/processed/users_metadata.json", 'r') as f:
    metadata = json.load(f)

users_data = {}
for username in metadata['users']:
    df = pd.read_json(f"data/processed/{username}_data.jsonl", lines=True)
    users_data[username] = preprocessor.filter_quality(df)

# Load all comments for context
with jsonlines.open("data/raw/comments.jsonl") as reader:
    all_comments = list(reader)

# Create multi-user training pairs
pairs = preprocessor.create_multi_user_training_pairs(users_data, all_comments)

# Format for training (username is always included per-example)
formatted = preprocessor.format_for_training(
    pairs,
    format_type="chatml",
)

# Split and save
train, val = preprocessor.split_data(formatted)
preprocessor.save_training_data(train, val)
```

### 2. Train the Model

Training works the same way, but now the model learns multiple personalities:

```bash
python -m litigpt.training.trainer
```

The system prompts will vary (in Italian):
- `"Sei alice_tech, un utente di Reddit. Rispondi nello stile e nel tono di scrittura di alice_tech."`
- `"Sei bob_gaming, un utente di Reddit. Rispondi nello stile e nel tono di scrittura di bob_gaming."`
- etc.

### 3. Build User Classifier

After training, build the classifier for automatic user selection:

```python
from litigpt.inference.classifier import build_user_classifier_from_data

# Build classifier
classifier = build_user_classifier_from_data("data/processed")

# Save for deployment
classifier.save_profiles("models/user_classifier.pkl")
```

Or use the pipeline:

```bash
python -c "
from litigpt.inference.classifier import build_user_classifier_from_data
classifier = build_user_classifier_from_data('data/processed')
classifier.save_profiles('models/user_classifier.pkl')
"
```

---

## User Selection Methods

### Method 1: TF-IDF Classification (Recommended)

**Best for:** General use, works well across diverse users

```python
from litigpt.inference.classifier import UserClassifier

classifier = UserClassifier()
classifier.load_profiles("models/user_classifier.pkl")

context = "I've been working on a neural network using PyTorch"
results = classifier.classify_context(context, top_k=3)

print("Top matches:")
for username, score in results:
    print(f"  {username}: {score:.3f}")
```

**Output:**
```
Top matches:
  alice_tech: 0.847
  bob_gaming: 0.234
  charlie_fitness: 0.198
```

### Method 2: Keyword Matching

**Best for:** Topic-specific routing, clear user domains

```python
from litigpt.inference.classifier import KeywordUserSelector

selector = KeywordUserSelector({
    'alice_tech': ['python', 'javascript', 'api', 'coding', 'ml'],
    'bob_gaming': ['valorant', 'league', 'fps', 'rank', 'esports'],
    'charlie_fitness': ['gym', 'workout', 'protein', 'cardio', 'pr']
})

context = "Just hit a new PR on deadlifts! 405 lbs!"
user = selector.predict_user(context)
print(f"Selected: {user}")  # charlie_fitness
```

### Method 3: Hybrid Approach

**Best for:** Maximum accuracy, combines both methods

```python
from litigpt.inference.classifier import HybridUserSelector

hybrid = HybridUserSelector(
    tfidf_classifier=classifier,
    keyword_selector=selector,
    default_user='alice_tech'
)

result = hybrid.predict_user(context, method='hybrid')
print(f"Selected: {result['username']}")
print(f"Method: {result['method']}")
print(f"Confidence: {result['confidence']:.3f}")
```

---

## Deployment

### Single Response

```python
from litigpt.inference.generator import RedditBotInference

bot = RedditBotInference(
    model_path="models/reddit_bot_lora",
    base_model="meta-llama/Llama-3.1-8B-Instruct"
)

# Specify user explicitly
response = bot.generate_as_user(
    context="What's the best way to learn Python?",
    username="alice_tech"
)

print(f"As alice_tech: {response}")
```

### Automatic User Selection

```python
from litigpt.inference.classifier import UserClassifier

# Load classifier
classifier = UserClassifier()
classifier.load_profiles("models/user_classifier.pkl")

# Get context
context = "What's the best way to learn Python?"

# Auto-select user
predicted_user = classifier.predict_user(context)

# Generate response
response = bot.generate_as_user(
    context=context,
    username=predicted_user
)

print(f"Selected {predicted_user}: {response}")
```

### Full Bot Deployment

```python
from litigpt.deployment.reddit_bot import RedditBot

bot = RedditBot(
    model_path="models/reddit_bot_lora",
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    subreddit_name="test",
    bot_username="multi_personality_bot",
    available_users=["alice_tech", "bob_gaming", "charlie_fitness"],
    user_classifier_path="models/user_classifier.pkl",
    reply_probability=0.2,
)

bot.run()
```

The bot will:
1. Monitor the subreddit
2. For each comment, analyze context
3. Auto-select which user to impersonate
4. Generate response in that user's style
5. Include disclaimer: "I'm a bot mimicking alice_tech's style"

---

## Examples

### Example 1: Tech Discussion

**Context:**
```
user1: What's the best framework for building REST APIs?
user2: I've been using Flask but considering FastAPI
```

**Bot selects:** `alice_tech` (high TF-IDF match on "framework", "APIs", "FastAPI")

**Response (as alice_tech):**
```
FastAPI is awesome! The automatic API documentation and built-in validation 
with Pydantic are game changers. Plus it's way faster than Flask for async stuff.
```

### Example 2: Gaming Discussion

**Context:**
```
user1: Anyone else hardstuck in Diamond?
user2: Yeah, keep getting bad teammates in ranked
```

**Bot selects:** `bob_gaming` (keyword match on "Diamond", "ranked")

**Response (as bob_gaming):**
```
Bro I feel you. Diamond is such an elo hell. Just focus on your own game 
and you'll climb. Maybe find a duo to queue with?
```

### Example 3: Fitness Discussion

**Context:**
```
user1: What's your bench press max?
user2: Just hit 225 for the first time!
```

**Bot selects:** `charlie_fitness` (keyword match on "bench press")

**Response (as charlie_fitness):**
```
Nice! 225 is a solid milestone. Make sure you're eating enough protein 
and getting rest days. Push for 245 next!
```

---

## Interactive Testing

Test different users interactively:

```python
from litigpt.inference.generator import RedditBotInference

bot = RedditBotInference(
    model_path="models/reddit_bot_lora",
    base_model="meta-llama/Llama-3.1-8B-Instruct"
)

# Interactive mode with user selection
bot.interactive_mode(available_users=["alice_tech", "bob_gaming", "charlie_fitness"])
```

**Usage:**
```
Context: @alice_tech What's your opinion on TypeScript?
As user: alice_tech
Bot: [response in alice's style]

Context: @bob_gaming Best agent for beginners in Valorant?
As user: bob_gaming
Bot: [response in bob's style]
```

---

## Best Practices

### 1. User Selection

- **Choose diverse users**: Different topics, writing styles, personalities
- **Minimum 100-200 comments per user**: More data = better style learning
- **Balanced dataset**: Similar amounts of data per user
- **Clear distinctions**: Users should have different topics/styles

### 2. Quality Control

```python
# Check user profile separation
from litigpt.inference.classifier import UserClassifier

classifier = UserClassifier()
classifier.load_profiles("models/user_classifier.pkl")

# Test on known user samples
for username in ["alice_tech", "bob_gaming", "charlie_fitness"]:
    test_file = f"data/processed/{username}_data.jsonl"
    df = pd.read_json(test_file, lines=True)
    
    # Sample some comments
    samples = df['body'].head(10).tolist()
    
    for sample in samples:
        predicted = classifier.predict_user(sample)
        print(f"Actual: {username}, Predicted: {predicted}")
```

### 3. Monitoring

Add logging to track which user is selected:

```python
# In deployment
logging.info(f"Context: {context[:100]}")
logging.info(f"Selected user: {predicted_user}")
logging.info(f"Confidence: {confidence:.3f}")
logging.info(f"Response: {response[:100]}")
```

### 4. Fallback Strategy

Always have a default user:

```yaml
bot:
  default_user: "alice_tech"  # Fallback if classifier unsure
```

---

## Troubleshooting

### Users Not Differentiating

**Problem:** Model responds similarly regardless of selected user

**Solutions:**
1. Increase training epochs (5-7 instead of 3)
2. Ensure users have distinct styles in training data
3. Check system prompts are being applied correctly
4. Use higher learning rate (3e-4) for stronger differentiation

### Classifier Selecting Wrong User

**Problem:** User selection doesn't match context

**Solutions:**
1. Build better classifier with more training data
2. Use keyword method for clear topic separation
3. Manually review and add keywords for each user
4. Combine TF-IDF + keywords in hybrid mode

### Low Confidence Scores

**Problem:** All similarity scores below 0.2

**Solutions:**
1. Lower the threshold in `predict_user(threshold=0.05)`
2. Set a default user for uncertain cases
3. Add more diverse training data per user

---

## Advanced: Custom Selection Logic

Implement your own selection logic:

```python
def custom_user_selector(context: str) -> str:
    """Custom logic for user selection"""
    
    context_lower = context.lower()
    
    # Time-based selection
    from datetime import datetime
    hour = datetime.now().hour
    
    if 9 <= hour <= 17:  # Work hours
        return "alice_tech"
    elif 18 <= hour <= 23:  # Evening
        return "bob_gaming"
    else:  # Late night
        return "charlie_fitness"
    
    # Or sentiment-based
    if any(word in context_lower for word in ['help', 'question', 'how']):
        return "alice_tech"  # Helpful user
    elif any(word in context_lower for word in ['lol', 'lmao', 'haha']):
        return "bob_gaming"  # Casual user
    
    return "alice_tech"  # Default
```

---

## Summary

Multi-user mode enables:
- ✅ Single model, multiple personalities
- ✅ Automatic user selection based on context
- ✅ More diverse and engaging responses
- ✅ Better coverage of different topics
- ✅ Efficient use of training resources

**Recommended workflow:**
1. Select 3-5 diverse users (different topics/styles)
2. Extract 200+ comments per user
3. Train model with multi-user flag enabled
4. Build TF-IDF classifier
5. Deploy with automatic user selection
6. Monitor and adjust based on performance

For questions or issues, see the main README troubleshooting section.
