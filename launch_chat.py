#!/usr/bin/env python3
"""
Quick Launch Script for Reddit Bot Chat Interfaces
Simplifies launching Gradio or Ollama interfaces
"""

import argparse
import subprocess
import sys
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    missing = []
    
    try:
        import gradio
    except ImportError:
        missing.append("gradio")
    
    try:
        import flask
    except ImportError:
        missing.append("flask")
    
    if missing:
        print(f"⚠️  Missing dependencies: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def launch_gradio(model_path, base_model, config, multi_user, share, port):
    """Launch Gradio interface"""
    cmd = [
        sys.executable,
        "litigpt/interface/gradio_app.py",
        "--model", model_path,
        "--base-model", base_model,
        "--config", config,
        "--port", str(port)
    ]
    
    if multi_user:
        cmd.append("--multi-user")
    
    if share:
        cmd.append("--share")
    
    print("\n" + "="*60)
    print("Launching Gradio Interface...")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Port: {port}")
    print(f"Multi-user: {multi_user}")
    print("="*60 + "\n")
    
    subprocess.run(cmd)

def launch_ollama(model_path, base_model, config, multi_user, host, port):
    """Launch Ollama-style interface"""
    cmd = [
        sys.executable,
        "litigpt/interface/ollama.py",
        "--model", model_path,
        "--base-model", base_model,
        "--config", config,
        "--host", host,
        "--port", str(port)
    ]
    
    if multi_user:
        cmd.append("--multi-user")
    
    print("\n" + "="*60)
    print("Launching Ollama-Style Interface...")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"URL: http://{host}:{port}")
    print(f"Multi-user: {multi_user}")
    print("="*60 + "\n")
    
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(
        description="Quick Launch Reddit Bot Chat Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch Gradio (default)
  python launch_chat.py --model models/reddit_bot_lora

  # Launch Ollama-style
  python launch_chat.py --interface ollama --model models/reddit_bot_lora

  # Multi-user with public sharing
  python launch_chat.py --model models/reddit_bot_lora --multi-user --share

  # Custom port
  python launch_chat.py --interface ollama --model models/reddit_bot_lora --port 8000
        """
    )
    
    parser.add_argument(
        "--interface",
        choices=["gradio", "ollama"],
        default="gradio",
        help="Interface type (default: gradio)"
    )
    
    parser.add_argument(
        "--model",
        required=True,
        help="Path to fine-tuned model directory"
    )
    
    parser.add_argument(
        "--base-model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model identifier (default: Llama-3.1-8B)"
    )
    
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Config file path (default: config.yaml)"
    )
    
    parser.add_argument(
        "--multi-user",
        action="store_true",
        help="Enable multi-user mode"
    )
    
    # Gradio-specific options
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link (Gradio only)"
    )
    
    # Ollama-specific options
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (Ollama only, default: 127.0.0.1)"
    )
    
    # Common options
    parser.add_argument(
        "--port",
        type=int,
        help="Server port (default: 7860 for Gradio, 5000 for Ollama)"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Error: Model not found at {args.model}")
        print("\nMake sure you've trained a model first:")
        print("  python -m litigpt.pipeline --step train")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Set default port if not specified
    if args.port is None:
        args.port = 7860 if args.interface == "gradio" else 5000
    
    # Launch appropriate interface
    if args.interface == "gradio":
        launch_gradio(
            args.model,
            args.base_model,
            args.config,
            args.multi_user,
            args.share,
            args.port
        )
    else:
        launch_ollama(
            args.model,
            args.base_model,
            args.config,
            args.multi_user,
            args.host,
            args.port
        )

if __name__ == "__main__":
    main()
