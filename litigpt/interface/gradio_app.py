"""
Module 9: Gradio Chat Interface
Simple web interface for chatting with the Reddit bot
"""

import logging
from pathlib import Path

import gradio as gr

from litigpt.config import Config, InferenceConfig
from litigpt.inference.generator import generate_reply
from litigpt.model_utils import (
    DEFAULT_BASE_MODEL,
    DEFAULT_USERNAME,
    load_model_and_tokenizer,
    resolve_available_users,
)
from litigpt.prompts import DEFAULT_HUMAN_HANDLE, build_system_prompt, render_thread

logger = logging.getLogger(__name__)

# Gradio 6 takes css on launch() rather than on the Blocks constructor.
CUSTOM_CSS = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.chat-message {
    padding: 10px;
    border-radius: 8px;
    margin: 5px 0;
}
"""

def message_text(content) -> str:
    """A chat message's text, in whatever shape Gradio hands it back.

    Gradio 6 returns each Chatbot message's content as a list of parts,
    [{"text": "ciao", "type": "text"}], even when it was given a string.
    Passed on as-is, render_thread wrote that list's repr into the thread, so
    every turn after the first reached the model in a shape it never saw --
    and on a retry it copied the shape into its reply.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text") or ""
    if isinstance(content, list):
        return "\n".join(t for t in map(message_text, content) if t)
    return ""


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

        self.available_users: list[str] = resolve_available_users(
            model_path, self.config.data.processed_dir
        )

    def generate_response(self,
                         message: str,
                         history: list[dict],
                         username: str | None = None,
                         settings: InferenceConfig | None = None) -> str:
        """Generate a response; settings default to the config's inference."""

        # Build prompt — always include username
        effective_user = username or (self.available_users[0] if self.available_users else DEFAULT_USERNAME)

        # Build conversation context as a Reddit thread, matching the training
        # format: every prior comment is prefixed with its author, and the
        # model's own past turns carry the persona's name rather than a
        # generic "assistant" label it never saw during fine-tuning.
        context = render_thread(
            [
                (
                    DEFAULT_HUMAN_HANDLE if turn.get("role") == "user" else effective_user,
                    message_text(turn.get("content")),
                )
                for turn in history or []
            ]
            + [(DEFAULT_HUMAN_HANDLE, message)]
        )
        system_prompt = build_system_prompt(effective_user)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ]
        s = settings or self.config.inference
        return generate_reply(
            self.model, self.tokenizer, messages,
            max_new_tokens=s.max_new_tokens,
            temperature=s.temperature,
            top_p=s.top_p,
            top_k=s.top_k,
            repetition_penalty=s.repetition_penalty,
            min_words=s.min_reply_words,
            candidates=s.reply_candidates,
        )

    @staticmethod
    def settings_from_controls(temperature, max_tokens, top_p, top_k,
                               repetition_penalty, min_words,
                               candidates) -> InferenceConfig:
        """The sampling sliders' values, in the order create_interface wires them."""
        return InferenceConfig(
            temperature=float(temperature),
            max_new_tokens=int(max_tokens),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            min_reply_words=int(min_words),
            reply_candidates=int(candidates),
        )

    def create_interface(self):
        """Create Gradio interface"""

        with gr.Blocks(title="Reddit Bot Chat") as demo:
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

                    # Every slider starts at the config's value, so the page
                    # opens on what the config tuned and reloading resets it.
                    inf = self.config.inference
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=inf.temperature,
                        step=0.05,
                        label="Temperature",
                        info="Higher = more creative",
                    )

                    min_words = gr.Slider(
                        minimum=0,
                        maximum=30,
                        value=inf.min_reply_words,
                        step=1,
                        label="Min reply words",
                        info="Shorter replies are resampled; 0 turns it off",
                    )

                    max_tokens = gr.Slider(
                        minimum=32,
                        maximum=512,
                        value=inf.max_new_tokens,
                        step=16,
                        label="Max Tokens",
                        info="Maximum response length",
                    )

                    with gr.Accordion("More sampling", open=False):
                        candidates = gr.Slider(
                            minimum=1,
                            maximum=8,
                            value=inf.reply_candidates,
                            step=1,
                            label="Reply candidates",
                            info="Sampled at once; the first over the word "
                                 "floor wins. Unused when the floor is 0",
                        )
                        top_p = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=inf.top_p,
                            step=0.05,
                            label="Top-p",
                        )
                        top_k = gr.Slider(
                            minimum=0,
                            maximum=200,
                            value=inf.top_k,
                            step=5,
                            label="Top-k",
                            info="0 turns it off",
                        )
                        repetition_penalty = gr.Slider(
                            minimum=1.0,
                            maximum=1.5,
                            value=inf.repetition_penalty,
                            step=0.05,
                            label="Repetition penalty",
                        )

                    # Order matches settings_from_controls.
                    sampling = [temperature, max_tokens, top_p, top_k,
                                repetition_penalty, min_words, candidates]

                    gr.Markdown("### Info")
                    info_text = f"""
                    **Model**: {self.config.model.base_model}

                    **Users**: {len(self.available_users) or 1}
                    """
                    gr.Markdown(info_text)

            # Examples
            gr.Examples(
                examples=[
                    ["Benvenuto su litigi!"],
                    ["Mi hanno bannato da r/italy"],
                    ["Chi è talpa?"],
                ],
                inputs=msg,
                label="Example Questions",
            )

            # Event handlers
            def respond(message, chat_history, username, *controls):
                chat_history = chat_history or []
                if not message.strip():
                    return "", chat_history

                response = self.generate_response(
                    message,
                    chat_history,
                    username or None,
                    self.settings_from_controls(*controls),
                )

                return "", chat_history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response},
                ]

            def retry_last(chat_history, username, *controls):
                chat_history = chat_history or []

                # Walk back to the last user message, dropping it and anything
                # after it, then answer it again.
                last_user = next(
                    (i for i in range(len(chat_history) - 1, -1, -1)
                     if chat_history[i].get("role") == "user"),
                    None,
                )
                if last_user is None:
                    return chat_history

                last_message = message_text(chat_history[last_user].get("content"))
                chat_history = chat_history[:last_user]

                response = self.generate_response(
                    last_message,
                    chat_history,
                    username or None,
                    self.settings_from_controls(*controls),
                )

                return chat_history + [
                    {"role": "user", "content": last_message},
                    {"role": "assistant", "content": response},
                ]

            # Wire up events
            msg.submit(
                respond,
                [msg, chatbot, username_selector, *sampling],
                [msg, chatbot],
            )

            submit.click(
                respond,
                [msg, chatbot, username_selector, *sampling],
                [msg, chatbot],
            )

            clear.click(list, None, chatbot, queue=False)

            retry.click(
                retry_last,
                [chatbot, username_selector, *sampling],
                chatbot,
            )

        return demo

    def launch(self, share: bool = False, server_port: int = 7860):
        """Launch the Gradio interface"""
        demo = self.create_interface()
        demo.launch(
            share=share,
            server_port=server_port,
            css=CUSTOM_CSS,
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
