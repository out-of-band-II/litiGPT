"""
Module 4: Model Inference
Generate responses using the fine-tuned model

generate_reply is the one place a reply is sampled. RedditBotInference (the
blind evaluation, the Reddit bot, the pipeline's eval step), the Gradio app
and the Ollama server all call it, so a sampling rule such as the short-reply
floor applies everywhere or nowhere.
"""

import logging

import torch

from litigpt.config import InferenceConfig
from litigpt.model_utils import (
    DEFAULT_BASE_MODEL,
    DEFAULT_USERNAME,
    load_model_and_tokenizer,
)
from litigpt.prompts import build_system_prompt, count_words

logger = logging.getLogger(__name__)


def pick_reply(candidates: list[str], min_words: int) -> str:
    """
    The first candidate with at least min_words words, else the longest.

    First, not longest: candidates are independent samples, so taking the
    first that qualifies is rejection sampling -- a reply drawn from what the
    model would say, given that it says something. Taking the longest would
    push every persona toward its most verbose register.
    """
    replies = [c.strip() for c in candidates]
    if not replies:
        return ""
    for reply in replies:
        if count_words(reply) >= min_words:
            return reply
    return max(replies, key=count_words)


def generate_reply(model, tokenizer, messages: list[dict], *,
                   max_new_tokens: int,
                   temperature: float,
                   top_p: float,
                   top_k: int,
                   repetition_penalty: float,
                   min_words: int = 0,
                   candidates: int = 1) -> str:
    """
    Sample a reply to chat messages, resampling ones under min_words words.

    All candidates are decoded in one batch with num_return_sequences, so on
    a GPU the floor costs one generation's latency, not one per retry.
    With min_words 0 or a single candidate this is a plain sampled reply.
    """
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)

    n = max(1, candidates) if min_words > 0 else 1
    pad_token_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=n,
            pad_token_id=pad_token_id,
        )

    start = inputs["input_ids"].shape[1]
    texts = [tokenizer.decode(seq[start:], skip_special_tokens=True) for seq in outputs]
    if n > 1:
        short = sum(count_words(t.strip()) < min_words for t in texts)
        if short:
            logger.info("%d of %d candidates were under %d words", short, n, min_words)
    return pick_reply(texts, min_words if n > 1 else 0)


class RedditBotInference:
    def __init__(self,
                 model_path: str,
                 base_model: str = DEFAULT_BASE_MODEL,
                 use_lora: bool = True,
                 load_in_4bit: bool = True,
                 inference: InferenceConfig | None = None):
        """
        Initialize inference engine

        Args:
            model_path: Path to fine-tuned model or LoRA adapters
            base_model: Base model name (if using LoRA)
            use_lora: Whether to load LoRA adapters
            load_in_4bit: Whether to use 4-bit quantization
            inference: Sampling settings; any argument generate_response is
                not given comes from here.
        """
        self.model_path = model_path
        self.base_model = base_model
        self.inference = inference or InferenceConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and tokenizer
        self.model, self.tokenizer, _ = load_model_and_tokenizer(
            base_model=base_model,
            adapter_path=model_path if use_lora else None,
            load_in_4bit=load_in_4bit,
        )

    def format_prompt(self, context: str,
                     system_prompt: str | None = None,
                     username: str | None = None) -> list[dict]:
        """
        Format conversation context into chat messages

        Args:
            context: The conversation context
            system_prompt: Optional system prompt override
            username: Username to impersonate (for multi-user models)
        """
        if system_prompt is None:
            if username:
                system_prompt = build_system_prompt(username)
            else:
                system_prompt = build_system_prompt(DEFAULT_USERNAME)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ]

    def generate_response(self,
                         context: str,
                         system_prompt: str | None = None,
                         username: str | None = None,
                         max_new_tokens: int | None = None,
                         temperature: float | None = None,
                         top_p: float | None = None,
                         top_k: int | None = None,
                         repetition_penalty: float | None = None) -> str:
        """
        Generate a response given context

        Args:
            context: The conversation context
            system_prompt: Optional system prompt override
            username: Username to impersonate (for multi-user models)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens

        Unset sampling arguments come from self.inference, as do the
        short-reply floor settings.
        """
        cfg = self.inference
        return generate_reply(
            self.model, self.tokenizer,
            self.format_prompt(context, system_prompt, username),
            max_new_tokens=max_new_tokens if max_new_tokens is not None else cfg.max_new_tokens,
            temperature=temperature if temperature is not None else cfg.temperature,
            top_p=top_p if top_p is not None else cfg.top_p,
            top_k=top_k if top_k is not None else cfg.top_k,
            repetition_penalty=(repetition_penalty if repetition_penalty is not None
                                else cfg.repetition_penalty),
            min_words=cfg.min_reply_words,
            candidates=cfg.reply_candidates,
        )

    def generate_as_user(self, context: str, username: str, **kwargs) -> str:
        """
        Convenience method to generate response as specific user

        Args:
            context: Conversation context
            username: Username to impersonate
            **kwargs: Additional generation parameters
        """
        return self.generate_response(context, username=username, **kwargs)

    def interactive_mode(self, available_users: list[str] | None = None):
        """
        Run interactive testing mode

        Args:
            available_users: List of users the model can impersonate
        """

        print("\n=== Interactive Mode ===")

        if available_users:
            print(f"Available users: {', '.join(available_users)}")
            print("You can specify a user with: @username context")

        print("Enter conversation context (or 'quit' to exit)")
        print("=" * 50)

        while True:
            user_input = input("\nContext: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if not user_input:
                continue

            # Parse username if provided
            username = None
            context = user_input

            if user_input.startswith('@') and ' ' in user_input:
                parts = user_input.split(' ', 1)
                username = parts[0][1:]  # Remove @
                context = parts[1]

                if available_users and username not in available_users:
                    print(f"Warning: {username} not in trained users: {available_users}")

            print("\nGenerating response...")

            if username:
                print(f"As user: {username}")
                response = self.generate_as_user(context, username)
            else:
                response = self.generate_response(context)

            print(f"\nBot: {response}")
            print("-" * 50)

if __name__ == "__main__":
    # Example usage with LoRA adapters
    bot = RedditBotInference(
        model_path="models/reddit_bot_lora",
        base_model=DEFAULT_BASE_MODEL,
        use_lora=True,
        load_in_4bit=True
    )

    # Test generation - single user mode
    context = """user1: What's your favorite programming language?
user2: I've been using Python a lot lately, but I'm curious about Rust."""

    response = bot.generate_response(context, temperature=0.8)
    print(f"Context:\n{context}\n")
    print(f"Bot Response:\n{response}\n")

    # Test generation - multi-user mode
    print("="*60)
    print("Multi-User Mode Example")
    print("="*60)

    # List of users the model was trained on
    available_users = ["alice", "bob", "charlie"]

    # Generate as different users
    for username in available_users:
        response = bot.generate_as_user(
            context=context,
            username=username,
            temperature=0.8
        )
        print(f"\nAs {username}: {response}")

    # Interactive mode with user selection
    # bot.interactive_mode(available_users=available_users)
