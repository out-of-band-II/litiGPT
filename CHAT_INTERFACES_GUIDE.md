# Chat Interfaces Installation & Usage Guide

## Overview

Two chat interface options are provided:

1. **Gradio Interface** - Simple, feature-rich, easy to deploy
2. **Ollama-Style Interface** - Modern, terminal-inspired design with streaming

---

## Installation

### Additional Dependencies

```bash
# For Gradio interface
pip install gradio

# For Ollama-style interface
pip install flask

# Both interfaces need these (should already be installed)
pip install torch transformers peft bitsandbytes pyyaml
```

---

## Option 1: Gradio Interface

### Features
- Clean, modern UI
- Built-in examples
- Adjustable temperature and max tokens
- Multi-user support
- Easy sharing via public link
- Mobile-friendly

### Launch Command

**Single-user bot:**
```bash
python -m litigpt.interface.gradio_app \
  --model models/reddit_bot_lora \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --config config.yaml
```

**Multi-user bot:**
```bash
python -m litigpt.interface.gradio_app \
  --model models/reddit_bot_lora \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --config config.yaml \
  --multi-user
```

**With public sharing:**
```bash
python -m litigpt.interface.gradio_app \
  --model models/reddit_bot_lora \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --share
```

**Custom port:**
```bash
python -m litigpt.interface.gradio_app \
  --model models/reddit_bot_lora \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --port 8080
```

### Access
- Local: http://localhost:7860
- With `--share`: Public URL will be displayed in terminal

### Screenshots

```
┌─────────────────────────────────────────────────────────────┐
│ 🤖 Reddit Bot Chat Interface                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ ┌─────────────────────────┐  ┌──────────────────┐          │
│ │  Conversation           │  │  Settings        │          │
│ │                         │  │                  │          │
│ │  User: What's your     │  │  Select User     │          │
│ │  favorite language?     │  │  [Dropdown ▼]    │          │
│ │                         │  │                  │          │
│ │  Bot: I really like    │  │  Temperature     │          │
│ │  Python for its...      │  │  [====o----] 0.8 │          │
│ │                         │  │                  │          │
│ └─────────────────────────┘  │  Max Tokens      │          │
│                              │  [====o----] 256 │          │
│ [Type message here...]       │                  │          │
│                              │  Info:           │          │
│ [Clear] [Retry] [Send]       │  Model: Llama-3.1│          │
│                              └──────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

---

## Option 2: Ollama-Style Interface

### Features
- Beautiful terminal-inspired UI
- Real-time streaming responses
- Typing indicators
- Message timestamps
- Dark theme
- Smooth animations
- Multi-user support

### Launch Command

**Basic:**
```bash
python -m litigpt.interface.ollama \
  --model models/reddit_bot_lora \
  --base-model meta-llama/Llama-3.1-8B-Instruct
```

**Multi-user:**
```bash
python -m litigpt.interface.ollama \
  --model models/reddit_bot_lora \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --multi-user
```

**Custom host/port:**
```bash
python -m litigpt.interface.ollama \
  --model models/reddit_bot_lora \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

### Access
- Local: http://localhost:5000
- Network: http://your-ip:5000 (if using --host 0.0.0.0)

### Design Preview

```
┌─────────────────────────────────────────────────────────────┐
│ 🤖 Reddit Bot Chat                            [Ready]       │
├─────────────────────────────────────────────────────────────┤
│ User: [user1 ▼]  Temp: [o----] 0.8  Tokens: [===o-] 256   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  👤  YOU                                    3:45 PM         │
│      What's your take on AI?                                │
│                                                              │
│  🤖  ASSISTANT                              3:45 PM         │
│      I think AI is fascinating! The recent developments...  │
│                                                              │
│  👤  YOU                                    3:46 PM         │
│      Tell me more                                           │
│                                                              │
│  Thinking • • •                                             │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│ [Type your message... (Shift+Enter for new line)]          │
│                                              [Send] [Clear] │
└─────────────────────────────────────────────────────────────┘
```

---

## Comparison

| Feature | Gradio | Ollama-Style |
|---------|--------|--------------|
| Setup | Very Easy | Easy |
| Design | Modern, clean | Terminal-inspired |
| Streaming | No | Yes |
| Examples | Built-in | Manual |
| Sharing | Public link | Manual setup |
| Mobile | Excellent | Good |
| Customization | Limited | Full control |
| Dependencies | 1 package | 1 package |

---

## Advanced Usage

### Multi-User Configuration

Both interfaces support multi-user mode, which requires:

1. **Trained multi-user model**
2. **User metadata file** at `data/processed/users_metadata.json`

The metadata file is automatically created when using `extract_multiple_users()` in Module 1.

### Custom Styling (Ollama-Style)

Edit the CSS in `litigpt/interface/ollama.py`:

```python
# Change colors
.header h1 {
    color: #00ff88;  # Change accent color
}

# Change background
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
```

### API Integration (Ollama-Style)

The Ollama interface exposes a REST API:

```bash
# Send message
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello!",
    "session_id": "my_session",
    "username": "user1",
    "temperature": 0.8,
    "max_tokens": 256
  }'

# Get history
curl http://localhost:5000/api/history/my_session

# Clear conversation
curl -X POST http://localhost:5000/api/clear/my_session

# Get bot info
curl http://localhost:5000/api/info
```

---

## Deployment Options

### Local Network Access

**Gradio:**
```bash
python -m litigpt.interface.gradio_app \
  --model models/reddit_bot_lora \
  --port 7860
```

**Ollama:**
```bash
python -m litigpt.interface.ollama \
  --model models/reddit_bot_lora \
  --host 0.0.0.0 \
  --port 5000
```

Access from other devices: `http://your-local-ip:port`

### Public Access

**Gradio (easiest):**
```bash
python -m litigpt.interface.gradio_app \
  --model models/reddit_bot_lora \
  --share
```

**Ollama (requires tunneling):**
```bash
# Using ngrok
ngrok http 5000

# Using cloudflared
cloudflared tunnel --url localhost:5000
```

### Docker Deployment

Create `Dockerfile.chat`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install torch transformers peft bitsandbytes \
    flask gradio pyyaml

# Copy files
COPY litigpt/interface/gradio_app.py .
COPY litigpt/interface/ollama.py .
COPY config.yaml .
COPY models/ models/

# Expose ports
EXPOSE 7860 5000

# Default to Gradio
CMD ["python", "-m", "litigpt.interface.gradio_app", \
     "--model", "models/reddit_bot_lora", \
     "--port", "7860"]
```

Build and run:
```bash
docker build -f Dockerfile.chat -t reddit-bot-chat .

# Gradio
docker run -p 7860:7860 reddit-bot-chat

# Ollama
docker run -p 5000:5000 reddit-bot-chat \
  python -m litigpt.interface.ollama \
  --model models/reddit_bot_lora \
  --host 0.0.0.0
```

---

## Troubleshooting

### Model Loading Errors

**Out of Memory:**
```bash
# Use smaller batch size during loading
# Or use CPU (slow)
CUDA_VISIBLE_DEVICES="" python -m litigpt.interface.gradio_app ...
```

**Model not found:**
```bash
# Check model path
ls models/reddit_bot_lora/

# Should contain:
# - adapter_config.json
# - adapter_model.safetensors
```

### Port Already in Use

**Gradio:**
```bash
python -m litigpt.interface.gradio_app --port 7861
```

**Ollama:**
```bash
python -m litigpt.interface.ollama --port 5001
```

### Slow Response Generation

- Reduce `max_tokens`
- Use smaller model (Phi-3 instead of Llama)
- Enable GPU if available
- Consider using vLLM for faster inference

### Interface Not Accessible

**Check firewall:**
```bash
# Linux
sudo ufw allow 7860
sudo ufw allow 5000

# macOS
# Allow in System Preferences > Security
```

**Check if server is running:**
```bash
netstat -tulpn | grep 7860
netstat -tulpn | grep 5000
```

---

## Performance Tips

### GPU Optimization

```python
# In the interface files, adjust quantization:

# 8-bit (faster, more memory)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

# 4-bit (slower, less memory)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
```

### Batch Processing (Ollama)

For multiple simultaneous users:

```python
# In litigpt/interface/ollama.py
# Increase worker threads
if __name__ == "__main__":
    from werkzeug.serving import run_simple
    run_simple(args.host, args.port, app, threaded=True)
```

### Caching

Enable KV cache for faster repeated inference:

```python
# In generate function
outputs = self.model.generate(
    **inputs,
    use_cache=True,  # Enable KV cache
    ...
)
```

---

## Security Considerations

### Rate Limiting (Ollama)

Add to `litigpt/interface/ollama.py`:

```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    default_limits=["100 per hour"]
)

@app.route('/api/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat():
    # ...
```

### Authentication

Add basic auth:

```python
from functools import wraps
from flask import request, abort

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get('Authorization')
        if auth != 'Bearer YOUR_SECRET_TOKEN':
            abort(401)
        return f(*args, **kwargs)
    return decorated

@app.route('/api/chat', methods=['POST'])
@require_auth
def chat():
    # ...
```

---

## Next Steps

1. **Try both interfaces** to see which you prefer
2. **Customize the UI** to match your needs
3. **Deploy to production** using Docker or cloud services
4. **Add features** like conversation saving, user profiles, etc.

For production deployment, see Module 12 (Cloud Deployment Scripts).
