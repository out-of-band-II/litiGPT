"""
Module 4: Model Inference
Generate responses using the fine-tuned model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import List, Dict, Optional

class RedditBotInference:
    def __init__(self, 
                 model_path: str,
                 base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
                 use_lora: bool = True,
                 load_in_4bit: bool = True):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to fine-tuned model or LoRA adapters
            base_model: Base model name (if using LoRA)
            use_lora: Whether to load LoRA adapters
            load_in_4bit: Whether to use 4-bit quantization
        """
        self.model_path = model_path
        self.base_model = base_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model(use_lora, load_in_4bit)
        
    def _load_model(self, use_lora: bool, load_in_4bit: bool):
        """Load model and tokenizer"""
        
        print(f"Loading model from {self.model_path}")
        
        # Quantization config
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            bnb_config = None
        
        # Load base model
        if use_lora:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            # Load LoRA adapters
            model = PeftModel.from_pretrained(model, self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            # Load merged model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        
        print("Model loaded successfully")
        return model, tokenizer
    
    def format_prompt(self, context: str, 
                     system_prompt: Optional[str] = None,
                     username: Optional[str] = None) -> List[Dict]:
        """
        Format conversation context into chat messages
        
        Args:
            context: The conversation context
            system_prompt: Optional system prompt override
            username: Username to impersonate (for multi-user models)
        """
        
        if system_prompt is None:
            if username:
                system_prompt = f"You are {username}, a Reddit user. Respond in {username}'s writing style and tone."
            else:
                system_prompt = "You are a helpful Reddit user responding to comments in a conversational manner."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]
        
        return messages
    
    def generate_response(self,
                         context: str,
                         system_prompt: Optional[str] = None,
                         username: Optional[str] = None,
                         max_new_tokens: int = 256,
                         temperature: float = 0.7,
                         top_p: float = 0.9,
                         top_k: int = 50,
                         repetition_penalty: float = 1.1) -> str:
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
        """
        
        # Format prompt
        messages = self.format_prompt(context, system_prompt, username)
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def generate_as_user(self, context: str, username: str, **kwargs) -> str:
        """
        Convenience method to generate response as specific user
        
        Args:
            context: Conversation context
            username: Username to impersonate
            **kwargs: Additional generation parameters
        """
        return self.generate_response(context, username=username, **kwargs)
    
    def interactive_mode(self, available_users: Optional[List[str]] = None):
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

class RedditBotInferenceVLLM:
    """
    Alternative inference using vLLM for faster generation
    Requires: pip install vllm
    """
    def __init__(self, model_path: str):
        from vllm import LLM, SamplingParams
        
        self.model_path = model_path
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9
        )
        
    def generate_response(self,
                         context: str,
                         max_tokens: int = 256,
                         temperature: float = 0.7,
                         top_p: float = 0.9) -> str:
        from vllm import SamplingParams
        
        # Format prompt (assuming ChatML format)
        prompt = f"<|im_start|>system\nYou are a helpful Reddit user.<|im_end|>\n<|im_start|>user\n{context}<|im_end|>\n<|im_start|>assistant\n"
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

if __name__ == "__main__":
    # Example usage with LoRA adapters
    bot = RedditBotInference(
        model_path="models/reddit_bot_lora",
        base_model="meta-llama/Llama-3.1-8B-Instruct",
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