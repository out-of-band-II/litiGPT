"""
Module 9: Gradio Chat Interface
Simple web interface for chatting with the Reddit bot
"""

import logging

import gradio as gr
import torch
from pathlib import Path
from typing import List, Tuple, Optional
from litigpt.config import Config
from litigpt.prompts import build_system_prompt
from litigpt.model_utils import (
    load_model_and_tokenizer,
    load_user_metadata,
    DEFAULT_USERNAME,
    DEFAULT_BASE_MODEL,
)

logger = logging.getLogger(__name__)

class GradioChatInterface:
    def __init__(self,
                 model_path: str,
                 base_model: str,
                 config_path: str = "config.yaml"):
        """
        Initialize Gradio chat interface

        Args:
            model_path: Path to fine-tuned LoRA model
            base_model: Base model identifier
            config_path: Path to config file
        """
        self.model_path = Path(model_path)
        self.config = Config.from_yaml(config_path)

        # Load model
        logger.info("Loading model...")
        self.model, self.tokenizer, _ = load_model_and_tokenizer(
            base_model=base_model,
            adapter_path=model_path,
        )
        logger.info("Model loaded successfully!")

        # Load user metadata
        self.available_users: List[str] = load_user_metadata(
            self.config.data.processed_dir
        )

    def generate_response(self,
                         message: str,
                         history: List[Tuple[str, str]],
                         username: Optional[str] = None,
                         temperature: float = 0.8,
                         max_tokens: int = 256) -> str:
        """Generate response to user message"""

        # Build conversation context
        conversation = []
        for user_msg, bot_msg in history:
            conversation.append(f"user: {user_msg}")
            conversation.append(f"assistant: {bot_msg}")
        conversation.append(f"user: {message}")

        context = "\n".join(conversation)

        # Build prompt — always include username
        effective_user = username or (self.available_users[0] if self.available_users else DEFAULT_USERNAME)
        system_prompt = build_system_prompt(effective_user)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)

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
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return response.strip()

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
                # [Bot] Reddit Bot Chat Interface
                Chat with your fine-tuned Reddit bot!
                """
            )

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        height=500,
                        label="Conversation",
                        show_label=True,
                        bubble_full_width=False,
                    )

                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Type your message here...",
                            show_label=False,
                            scale=4,
                        )
                        submit = gr.Button("Send", variant="primary", scale=1)

                    with gr.Row():
                        clear = gr.Button("Clear Chat")
                        retry = gr.Button("Retry Last")

                with gr.Column(scale=1):
                    gr.Markdown("### Settings")

                    # User selector (always shown when users are available)
                    if self.available_users:
                        username_selector = gr.Dropdown(
                            choices=self.available_users,
                            value=self.available_users[0],
                            label="Select User to Mimic",
                            info="Choose which Reddit user's style to use",
                        )
                    else:
                        username_selector = gr.Textbox(visible=False)

                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.8,
                        step=0.1,
                        label="Temperature",
                        info="Higher = more creative",
                    )

                    max_tokens = gr.Slider(
                        minimum=50,
                        maximum=512,
                        value=256,
                        step=50,
                        label="Max Tokens",
                        info="Maximum response length",
                    )

                    gr.Markdown("### Info")
                    info_text = f"""
                    **Model**: {self.config.model.base_model}

                    **Users**: {len(self.available_users) or 1}
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
                label="Example Questions",
            )

            # Event handlers
            def respond(message, chat_history, username, temp, max_tok):
                if not message.strip():
                    return "", chat_history

                response = self.generate_response(
                    message,
                    chat_history,
                    username or None,
                    temp,
                    max_tok,
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
                    username or None,
                    temp,
                    max_tok,
                )

                chat_history.append((last_message, response))
                return chat_history

            # Wire up events
            msg.submit(
                respond,
                [msg, chatbot, username_selector, temperature, max_tokens],
                [msg, chatbot],
            )

            submit.click(
                respond,
                [msg, chatbot, username_selector, temperature, max_tokens],
                [msg, chatbot],
            )

            clear.click(lambda: None, None, chatbot, queue=False)

            retry.click(
                retry_last,
                [chatbot, username_selector, temperature, max_tokens],
                chatbot,
            )

        return demo

    def launch(self, share: bool = False, server_port: int = 7860):
        """Launch the Gradio interface"""
        demo = self.create_interface()
        demo.launch(
            share=share,
            server_port=server_port,
            show_error=True,
        )


def main():
    """Main function to launch interface"""
    import argparse

    parser = argparse.ArgumentParser(description="Launch Reddit Bot Chat Interface")
    parser.add_argument("--model", required=True, help="Path to fine-tuned model")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                       help="Base model identifier")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
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
    )

    print(f"\n{'='*60}")
    print("Starting Reddit Bot Chat Interface...")
    print(f"{'='*60}\n")

    interface.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
