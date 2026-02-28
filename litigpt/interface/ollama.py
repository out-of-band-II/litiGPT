"""
Module 10: Ollama-Style Chat Interface
Modern, terminal-inspired chat interface for Reddit bot
"""

from flask import Flask, render_template, request, jsonify, Response
from pathlib import Path
import yaml
import json
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datetime import datetime
import threading
import queue

app = Flask(__name__)

class OllamaChatInterface:
    def __init__(self,
                 model_path: str,
                 base_model: str,
                 config_path: str = "config.yaml",
                 multi_user: bool = False):
        """Initialize Ollama-style interface"""
        
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.multi_user = multi_user
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        print("Loading model...")
        self.tokenizer, self.model = self._load_model(base_model)
        print("Model loaded successfully!")
        
        # Load user metadata
        self.available_users = []
        if multi_user:
            self._load_user_metadata()
        
        # Conversation history
        self.conversations: Dict[str, List[Dict]] = {}
    
    def _load_model(self, base_model: str):
        """Load the fine-tuned model"""
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        model = PeftModel.from_pretrained(model, self.model_path)
        model.eval()
        
        return tokenizer, model
    
    def _load_user_metadata(self):
        """Load available users"""
        metadata_path = Path("data/processed/users_metadata.json")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.available_users = metadata.get('users', [])
    
    def generate_response_stream(self,
                                message: str,
                                session_id: str,
                                username: Optional[str] = None,
                                temperature: float = 0.8,
                                max_tokens: int = 256):
        """Generate response with streaming"""
        
        # Get conversation history
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        history = self.conversations[session_id]
        
        # Build context
        conversation = []
        for msg in history:
            if msg['role'] == 'user':
                conversation.append(f"user: {msg['content']}")
            else:
                conversation.append(f"assistant: {msg['content']}")
        conversation.append(f"user: {message}")
        
        context = "\n".join(conversation)
        
        # Build prompt
        if self.multi_user and username:
            system_prompt = f"You are {username}, a Reddit user. Respond in the style and tone of {username}."
        else:
            target_user = self.config['data'].get('target_username', 'a Reddit user')
            system_prompt = f"You are {target_user}, a Reddit user. Respond in your natural style."
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate with streaming
        from transformers import TextIteratorStreamer
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True
        )
        
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=streamer
        )
        
        # Start generation in thread
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream tokens
        full_response = ""
        for new_text in streamer:
            full_response += new_text
            yield new_text
        
        # Save to history
        self.conversations[session_id].append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        self.conversations[session_id].append({
            'role': 'assistant',
            'content': full_response,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_conversation(self, session_id: str) -> List[Dict]:
        """Get conversation history"""
        return self.conversations.get(session_id, [])
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history"""
        if session_id in self.conversations:
            self.conversations[session_id] = []


# Global interface instance
chat_interface = None


@app.route('/')
def index():
    """Render main chat interface"""
    return render_template('chat.html',
                          multi_user=chat_interface.multi_user,
                          available_users=chat_interface.available_users)


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat message"""
    data = request.json
    message = data.get('message', '')
    session_id = data.get('session_id', 'default')
    username = data.get('username')
    temperature = data.get('temperature', 0.8)
    max_tokens = data.get('max_tokens', 256)
    
    def generate():
        for chunk in chat_interface.generate_response_stream(
            message,
            session_id,
            username,
            temperature,
            max_tokens
        ):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        
        yield f"data: {json.dumps({'done': True})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id):
    """Get conversation history"""
    history = chat_interface.get_conversation(session_id)
    return jsonify({'history': history})


@app.route('/api/clear/<session_id>', methods=['POST'])
def clear_history(session_id):
    """Clear conversation"""
    chat_interface.clear_conversation(session_id)
    return jsonify({'status': 'success'})


@app.route('/api/info', methods=['GET'])
def get_info():
    """Get bot info"""
    return jsonify({
        'model': chat_interface.config['model']['base_model'],
        'multi_user': chat_interface.multi_user,
        'available_users': chat_interface.available_users,
        'target_user': chat_interface.config['data'].get('target_username')
    })


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reddit Bot Chat</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Monaco', 'Courier New', monospace;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 1200px;
            height: 90vh;
            background: #1e1e1e;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .header {
            background: #2d2d2d;
            padding: 20px;
            border-bottom: 2px solid #3d3d3d;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header h1 {
            color: #00ff88;
            font-size: 24px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .header .status {
            color: #888;
            font-size: 12px;
            background: #252525;
            padding: 5px 15px;
            border-radius: 20px;
        }

        .settings-bar {
            background: #252525;
            padding: 15px 20px;
            border-bottom: 1px solid #3d3d3d;
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }

        .settings-bar label {
            color: #aaa;
            font-size: 12px;
        }

        .settings-bar select,
        .settings-bar input {
            background: #1e1e1e;
            border: 1px solid #3d3d3d;
            color: #fff;
            padding: 5px 10px;
            border-radius: 4px;
            font-family: inherit;
            font-size: 12px;
        }

        .chat-area {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            scroll-behavior: smooth;
        }

        .message {
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            animation: fadeIn 0.3s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            flex-shrink: 0;
        }

        .message.user .avatar {
            background: #667eea;
        }

        .message.assistant .avatar {
            background: #00ff88;
        }

        .message .content {
            flex: 1;
        }

        .message .role {
            color: #888;
            font-size: 12px;
            margin-bottom: 5px;
            text-transform: uppercase;
        }

        .message.user .role {
            color: #667eea;
        }

        .message.assistant .role {
            color: #00ff88;
        }

        .message .text {
            color: #ddd;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .message .timestamp {
            color: #666;
            font-size: 10px;
            margin-top: 5px;
        }

        .input-area {
            background: #2d2d2d;
            padding: 20px;
            border-top: 2px solid #3d3d3d;
        }

        .input-container {
            display: flex;
            gap: 10px;
        }

        #messageInput {
            flex: 1;
            background: #1e1e1e;
            border: 2px solid #3d3d3d;
            color: #fff;
            padding: 15px;
            border-radius: 8px;
            font-family: inherit;
            font-size: 14px;
            resize: none;
            outline: none;
            transition: border-color 0.3s;
        }

        #messageInput:focus {
            border-color: #00ff88;
        }

        button {
            background: #00ff88;
            border: none;
            color: #1e1e1e;
            padding: 15px 30px;
            border-radius: 8px;
            font-family: inherit;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }

        button:hover {
            background: #00dd77;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,255,136,0.3);
        }

        button:active {
            transform: translateY(0);
        }

        button.secondary {
            background: #3d3d3d;
            color: #fff;
        }

        button.secondary:hover {
            background: #4d4d4d;
        }

        .typing-indicator {
            display: none;
            align-items: center;
            gap: 5px;
            color: #00ff88;
            font-size: 14px;
            margin: 20px 0;
        }

        .typing-indicator.active {
            display: flex;
        }

        .typing-indicator .dot {
            width: 8px;
            height: 8px;
            background: #00ff88;
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }

        .typing-indicator .dot:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing-indicator .dot:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.7; }
            30% { transform: translateY(-10px); opacity: 1; }
        }

        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #1e1e1e;
        }

        ::-webkit-scrollbar-thumb {
            background: #3d3d3d;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #4d4d4d;
        }

        .button-group {
            display: flex;
            gap: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>
                <span>🤖</span>
                <span>Reddit Bot Chat</span>
            </h1>
            <div class="status" id="status">Ready</div>
        </div>

        {% if multi_user %}
        <div class="settings-bar">
            <label>
                User:
                <select id="userSelect">
                    {% for user in available_users %}
                    <option value="{{ user }}">{{ user }}</option>
                    {% endfor %}
                </select>
            </label>
            <label>
                Temperature:
                <input type="range" id="temperature" min="0.1" max="1.5" step="0.1" value="0.8">
                <span id="tempValue">0.8</span>
            </label>
            <label>
                Max Tokens:
                <input type="range" id="maxTokens" min="50" max="512" step="50" value="256">
                <span id="tokensValue">256</span>
            </label>
        </div>
        {% endif %}

        <div class="chat-area" id="chatArea">
            <div class="message assistant">
                <div class="avatar">🤖</div>
                <div class="content">
                    <div class="role">Assistant</div>
                    <div class="text">Hi! I'm your Reddit bot. Ask me anything!</div>
                </div>
            </div>
        </div>

        <div class="typing-indicator" id="typingIndicator">
            <span>Thinking</span>
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>

        <div class="input-area">
            <div class="input-container">
                <textarea 
                    id="messageInput" 
                    placeholder="Type your message... (Shift+Enter for new line)"
                    rows="1"
                ></textarea>
                <div class="button-group">
                    <button id="sendButton">Send</button>
                    <button class="secondary" id="clearButton">Clear</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        const chatArea = document.getElementById('chatArea');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const clearButton = document.getElementById('clearButton');
        const typingIndicator = document.getElementById('typingIndicator');
        const status = document.getElementById('status');
        const sessionId = 'session_' + Date.now();

        // Settings
        const userSelect = document.getElementById('userSelect');
        const temperature = document.getElementById('temperature');
        const maxTokens = document.getElementById('maxTokens');
        const tempValue = document.getElementById('tempValue');
        const tokensValue = document.getElementById('tokensValue');

        if (temperature) {
            temperature.addEventListener('input', (e) => {
                tempValue.textContent = e.target.value;
            });
        }

        if (maxTokens) {
            maxTokens.addEventListener('input', (e) => {
                tokensValue.textContent = e.target.value;
            });
        }

        function addMessage(role, text) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            
            const avatar = role === 'user' ? '👤' : '🤖';
            const roleText = role === 'user' ? 'You' : 'Assistant';
            
            messageDiv.innerHTML = `
                <div class="avatar">${avatar}</div>
                <div class="content">
                    <div class="role">${roleText}</div>
                    <div class="text">${text}</div>
                    <div class="timestamp">${new Date().toLocaleTimeString()}</div>
                </div>
            `;
            
            chatArea.insertBefore(messageDiv, typingIndicator);
            chatArea.scrollTop = chatArea.scrollHeight;
            
            return messageDiv;
        }

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;

            // Add user message
            addMessage('user', message);
            messageInput.value = '';
            
            // Show typing indicator
            typingIndicator.classList.add('active');
            status.textContent = 'Thinking...';

            // Prepare request
            const requestData = {
                message: message,
                session_id: sessionId,
                username: userSelect ? userSelect.value : null,
                temperature: temperature ? parseFloat(temperature.value) : 0.8,
                max_tokens: maxTokens ? parseInt(maxTokens.value) : 256
            };

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(requestData)
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                
                let assistantMessage = addMessage('assistant', '');
                let fullText = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\n');

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = JSON.parse(line.slice(6));
                            
                            if (data.chunk) {
                                fullText += data.chunk;
                                assistantMessage.querySelector('.text').textContent = fullText;
                                chatArea.scrollTop = chatArea.scrollHeight;
                            }
                            
                            if (data.done) {
                                typingIndicator.classList.remove('active');
                                status.textContent = 'Ready';
                            }
                        }
                    }
                }
            } catch (error) {
                console.error('Error:', error);
                addMessage('assistant', 'Sorry, an error occurred.');
                typingIndicator.classList.remove('active');
                status.textContent = 'Error';
            }
        }

        // Event listeners
        sendButton.addEventListener('click', sendMessage);

        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        clearButton.addEventListener('click', async () => {
            if (confirm('Clear conversation history?')) {
                await fetch(`/api/clear/${sessionId}`, { method: 'POST' });
                chatArea.innerHTML = '';
                addMessage('assistant', 'Conversation cleared. How can I help you?');
            }
        });

        // Auto-resize textarea
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });
    </script>
</body>
</html>
"""


def save_template():
    """Save HTML template"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    template_path = templates_dir / "chat.html"
    with open(template_path, 'w') as f:
        f.write(HTML_TEMPLATE)
    
    print(f"Template saved to {template_path}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Ollama-Style Chat Interface")
    parser.add_argument("--model", required=True, help="Path to fine-tuned model")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Base model identifier")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--multi-user", action="store_true",
                       help="Enable multi-user mode")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    
    args = parser.parse_args()
    
    # Save template
    save_template()
    
    # Create interface
    global chat_interface
    chat_interface = OllamaChatInterface(
        model_path=args.model,
        base_model=args.base_model,
        config_path=args.config,
        multi_user=args.multi_user
    )
    
    print(f"\n{'='*60}")
    print("Reddit Bot Chat Interface")
    print(f"{'='*60}")
    print(f"\nServer starting on http://{args.host}:{args.port}")
    print(f"Press Ctrl+C to stop\n")
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
