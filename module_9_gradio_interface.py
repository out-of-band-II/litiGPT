"""
Module 9: Gradio Chat Interface
Simple web interface for chatting with the Reddit bot
"""

import gradio as gr
from pathlib import Path
import yaml
from typing import List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json

class GradioChatInterface:
    def __init__(self, 
                 model_path: str,
                 base_model: str,
                 config_path: str = "config.yaml",
                 multi_user: bool = False):
        """
        Initialize Gradio chat interface
        
        Args:
            model_path: Path to fine-tuned LoRA model
            base_model: Base model identifier
            config_path: Path to config file
            multi_user: Whether this is a multi-user bot
        """
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
        
        # Load user metadata if multi-user
        self.available_users = []
        if multi_user:
            self._load_user_metadata()
    
    def _load_model(self, base_model: str):
        """Load the fine-tuned model"""
        
        # Quantization config for efficient inference
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(model, self.model_path)
        model.eval()
        
        return tokenizer, model
    
    def _load_user_metadata(self):
        """Load available users from metadata"""
        metadata_path = Path("data/processed/users_metadata.json")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.available_users = metadata.get('users', [])
                print(f"Loaded {len(self.available_users)} users: {self.available_users}")
    
    def generate_response(self, 
                         message: str,
                         history: List[Tuple[str, str]],
                         username: Optional[str] = None,
                         temperature: float = 0.8,
                         max_tokens: int = 256) -> str:
        """
        Generate response to user message
        
        Args:
            message: User's message
            history: Chat history
            username: Username to mimic (for multi-user)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        
        # Build conversation context
        conversation = []
        for user_msg, bot_msg in history:
            conversation.append(f"user: {user_msg}")
            conversation.append(f"assistant: {bot_msg}")
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
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        return response
    
    def create_interface(self):
        """Create Gradio interface"""
        
        # Custom CSS
        css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .chat-message {
            padding: 10px;
            border-radius: 8px;
            margin: 5px 0;
        }
        """
        
        with gr.Blocks(css=css, title="Reddit Bot Chat") as demo:
            gr.Markdown(
                """
                # 🤖 Reddit Bot Chat Interface
                Chat with your fine-tuned Reddit bot!
                """
            )
            
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        height=500,
                        label="Conversation",
                        show_label=True,
                        bubble_full_width=False
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Type your message here...",
                            show_label=False,
                            scale=4
                        )
                        submit = gr.Button("Send", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear = gr.Button("Clear Chat")
                        retry = gr.Button("Retry Last")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Settings")
                    
                    # Multi-user selector
                    if self.multi_user and self.available_users:
                        username_selector = gr.Dropdown(
                            choices=self.available_users,
                            value=self.available_users[0] if self.available_users else None,
                            label="Select User to Mimic",
                            info="Choose which Reddit user's style to use"
                        )
                    else:
                        username_selector = gr.Textbox(visible=False)
                    
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.8,
                        step=0.1,
                        label="Temperature",
                        info="Higher = more creative"
                    )
                    
                    max_tokens = gr.Slider(
                        minimum=50,
                        maximum=512,
                        value=256,
                        step=50,
                        label="Max Tokens",
                        info="Maximum response length"
                    )
                    
                    gr.Markdown("### Info")
                    info_text = f"""
                    **Model**: {self.config['model']['base_model']}
                    
                    **Type**: {"Multi-user" if self.multi_user else "Single-user"}
                    
                    **Users**: {len(self.available_users) if self.multi_user else 1}
                    """
                    gr.Markdown(info_text)
            
            # Examples
            gr.Examples(
                examples=[
                    ["What do you think about AI?"],
                    ["What's your favorite programming language?"],
                    ["Tell me about your hobbies"],
                    ["What's the best pizza topping?"],
                ],
                inputs=msg,
                label="Example Questions"
            )
            
            # Event handlers
            def respond(message, chat_history, username, temp, max_tok):
                if not message.strip():
                    return "", chat_history
                
                response = self.generate_response(
                    message,
                    chat_history,
                    username if self.multi_user else None,
                    temp,
                    max_tok
                )
                
                chat_history.append((message, response))
                return "", chat_history
            
            def retry_last(chat_history, username, temp, max_tok):
                if not chat_history:
                    return chat_history
                
                last_message = chat_history[-1][0]
                chat_history = chat_history[:-1]
                
                response = self.generate_response(
                    last_message,
                    chat_history,
                    username if self.multi_user else None,
                    temp,
                    max_tok
                )
                
                chat_history.append((last_message, response))
                return chat_history
            
            # Wire up events
            msg.submit(
                respond,
                [msg, chatbot, username_selector, temperature, max_tokens],
                [msg, chatbot]
            )
            
            submit.click(
                respond,
                [msg, chatbot, username_selector, temperature, max_tokens],
                [msg, chatbot]
            )
            
            clear.click(lambda: None, None, chatbot, queue=False)
            
            retry.click(
                retry_last,
                [chatbot, username_selector, temperature, max_tokens],
                chatbot
            )
        
        return demo
    
    def launch(self, share: bool = False, server_port: int = 7860):
        """Launch the Gradio interface"""
        demo = self.create_interface()
        demo.launch(
            share=share,
            server_port=server_port,
            show_error=True
        )


def main():
    """Main function to launch interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Reddit Bot Chat Interface")
    parser.add_argument("--model", required=True, help="Path to fine-tuned model")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Base model identifier")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--multi-user", action="store_true", 
                       help="Enable multi-user mode")
    parser.add_argument("--share", action="store_true",
                       help="Create public share link")
    parser.add_argument("--port", type=int, default=7860,
                       help="Server port")
    
    args = parser.parse_args()
    
    # Create and launch interface
    interface = GradioChatInterface(
        model_path=args.model,
        base_model=args.base_model,
        config_path=args.config,
        multi_user=args.multi_user
    )
    
    print(f"\n{'='*60}")
    print("Starting Reddit Bot Chat Interface...")
    print(f"{'='*60}\n")
    
    interface.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
