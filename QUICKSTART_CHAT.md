# Reddit Bot Chat Interfaces - Quick Start

## What You Get

Two professional chat interfaces to interact with your Reddit bot:

1. **Gradio Interface** - Feature-rich, easy setup, shareable
2. **Ollama-Style Interface** - Beautiful, terminal-inspired, streaming responses

---

## Installation (30 seconds)

```bash
# Install chat interface dependencies
pip install gradio flask

# You should already have these from training
pip install torch transformers peft bitsandbytes pyyaml pydantic
```

---

## Launch (One Command)

### Quick Launch Script (Easiest)

```bash
# Gradio interface (recommended for first time)
python launch_chat.py --model models/reddit_bot_lora

# Ollama-style interface
python launch_chat.py --interface ollama --model models/reddit_bot_lora

# Public sharing (Gradio)
python launch_chat.py --model models/reddit_bot_lora --share
```

### Direct Launch

**Gradio:**
```bash
python -m litigpt.interface.gradio_app \
  --model models/reddit_bot_lora \
  --base-model meta-llama/Llama-3.1-8B-Instruct
```

**Ollama:**
```bash
python -m litigpt.interface.ollama \
  --model models/reddit_bot_lora \
  --base-model meta-llama/Llama-3.1-8B-Instruct
```

---

## Access

**Gradio:** http://localhost:7860
**Ollama:** http://localhost:5000

---

## Features Comparison

| Feature | Gradio | Ollama |
|---------|--------|--------|
| Setup Time | 1 minute | 1 minute |
| UI Style | Modern cards | Terminal |
| Streaming | No | Yes |
| Public Sharing | Yes (built-in) | Via ngrok |
| Mobile Support | Excellent | Good |
| Customization | Limited | Full CSS |

---

## Usage Examples

### Single User Bot

```bash
# Chat as the trained user
python launch_chat.py --model models/reddit_bot_lora
```

Then open your browser and start chatting!

### Multiple Users

If you trained on multiple users, the dropdown menu to select which user's style to use appears automatically.

### Public Demo

Share your bot with others:

```bash
python launch_chat.py --model models/reddit_bot_lora --share
```

You'll get a public URL like: `https://xxxxx.gradio.live`

### Custom Port

```bash
python launch_chat.py --model models/reddit_bot_lora --port 8080
```

---

## Keyboard Shortcuts

**Both Interfaces:**
- `Enter` - Send message
- `Shift+Enter` - New line

**Gradio:**
- Click examples to quick-fill

**Ollama:**
- Auto-scrolls to latest message

---

## Troubleshooting

### "Model not found"

Make sure you've trained a model first:
```bash
python run_pipeline.py --step train
```

### "Out of memory"

Use a smaller model or reduce max tokens in the UI.

### Port already in use

```bash
python launch_chat.py --model models/reddit_bot_lora --port 7861
```

### Slow responses

- First response is always slower (model loading)
- Subsequent responses are faster
- Reduce max_tokens in settings
- Consider using a smaller base model (Phi-3)

---

## Advanced Configuration

### Adjust Generation Settings

In the UI:
- **Temperature** (0.1-1.5): Higher = more creative
- **Max Tokens** (50-512): Maximum response length

### Custom System Prompt

System prompts are centralized in `litigpt/prompts.py`. To customize, edit `build_system_prompt()`:

```python
# litigpt/prompts.py
def build_system_prompt(username: str) -> str:
    return f"Sei {username}, un utente di Reddit. Rispondi nello stile e nel tono di scrittura di {username}."
```

### Add Authentication (Ollama)

```python
# In litigpt/interface/ollama.py
@app.before_request
def require_password():
    password = request.headers.get('X-Password')
    if password != 'your_secret_password':
        return jsonify({'error': 'Unauthorized'}), 401
```

---

## Integration with Your App

### REST API (Ollama Interface)

```python
import requests

response = requests.post('http://localhost:5000/api/chat', json={
    'message': 'Hello!',
    'session_id': 'user123',
    'temperature': 0.8,
    'max_tokens': 256
})
```

### Python Integration

```python
from litigpt.interface.gradio_app import GradioChatInterface

bot = GradioChatInterface(
    model_path="models/reddit_bot_lora",
    base_model="meta-llama/Llama-3.1-8B-Instruct"
)

response = bot.generate_response(
    message="What do you think?",
    history=[],
    temperature=0.8
)
```

---

## Deployment

### Docker

```dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements_with_chat.txt
CMD ["python", "launch_chat.py", "--model", "models/reddit_bot_lora"]
```

### Cloud Platforms

See `CHAT_INTERFACES_GUIDE.md` for:
- AWS deployment
- Google Cloud Run
- Heroku
- Railway

---

## What's Next?

1. Try both interfaces to see which you prefer
2. Customize the UI colors and styling
3. Add features like conversation saving
4. Deploy to production
5. Share with friends!

For more details, see `CHAT_INTERFACES_GUIDE.md`

---

## Support

If you have issues:
1. Check `CHAT_INTERFACES_GUIDE.md` troubleshooting section
2. Ensure model is trained and exists
3. Verify all dependencies are installed
4. Check console output for errors

Enjoy chatting with your Reddit bot! 🤖
