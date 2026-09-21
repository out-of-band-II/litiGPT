#!/usr/bin/env python3
"""
Quick Launch Script for Reddit Bot Chat Interfaces
Simplifies launching Gradio or Ollama interfaces
"""

import argparse
import subprocess
import sys
from pathlib import Path

from litigpt.model_utils import DEFAULT_BASE_MODEL

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
        print(f"Missing dependencies: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"pip install {' '.join(missing)}")
        return False

    return True

def launch_gradio(model_path, base_model, config, share, port):
    """Launch Gradio interface"""
    cmd = [
        sys.executable,
        "litigpt/interface/gradio_app.py",
        "--model", model_path,
        "--base-model", base_model,
        "--config", config,
        "--port", str(port),
    ]

    if share:
        cmd.append("--share")

    print("\n" + "="*60)
    print("Launching Gradio Interface...")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Port: {port}")
    print("="*60 + "\n")

    subprocess.run(cmd)

def launch_blind(model_path, base_model, config, share, port, host,
                 oracle, classifier, seed):
    """Launch the blind persona evaluation"""
    cmd = [
        sys.executable,
        "-m", "litigpt.interface.blind_eval",
        "--base-model", base_model,
        "--config", config,
        "--host", host,
        "--port", str(port),
    ]

    if model_path:
        cmd += ["--model", model_path]
    if oracle:
        cmd.append("--oracle")
    if classifier:
        cmd.append("--classifier")
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if share:
        cmd.append("--share")

    print("\n" + "="*60)
    print("Launching Blind Persona Evaluation...")
    print("="*60)
    print(f"Source: {'real held-out comments (ceiling)' if oracle else model_path}")
    print(f"Port: {port}")
    print("="*60 + "\n")

    subprocess.run(cmd)

def launch_ollama(model_path, base_model, config, host, port):
    """Launch Ollama-style interface"""
    cmd = [
        sys.executable,
        "litigpt/interface/ollama.py",
        "--model", model_path,
        "--base-model", base_model,
        "--config", config,
        "--host", host,
        "--port", str(port),
    ]

    print("\n" + "="*60)
    print("Launching Ollama-Style Interface...")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"URL: http://{host}:{port}")
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

  # Blind evaluation: guess which persona the model is impersonating
  python launch_chat.py --interface blind --model models/litigpt_top30_lora

  # Ceiling run first — real comments, no GPU needed, no adapter needed
  python launch_chat.py --interface blind --oracle

  # Launch Ollama-style
  python launch_chat.py --interface ollama --model models/reddit_bot_lora

  # With public sharing
  python launch_chat.py --model models/reddit_bot_lora --share

  # Custom port
  python launch_chat.py --interface ollama --model models/reddit_bot_lora --port 8000
        """
    )

    parser.add_argument(
        "--interface",
        choices=["gradio", "ollama", "blind"],
        default="gradio",
        help="Interface type (default: gradio)"
    )

    parser.add_argument(
        "--model",
        help="Path to fine-tuned model directory "
             "(optional only for --interface blind --oracle)"
    )

    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=f"Base model identifier (default: {DEFAULT_BASE_MODEL})"
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Config file path (default: config.yaml)"
    )

    # Gradio-specific options
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link (Gradio and blind eval)"
    )

    # Blind-evaluation options
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Blind eval only: answer with the persona's real held-out "
             "comments instead of generating. Establishes the ceiling your "
             "model scores should be read against; needs no GPU or adapter."
    )

    parser.add_argument(
        "--classifier",
        action="store_true",
        help="Blind eval only: have the TF-IDF attributor guess each round too"
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Blind eval only: seed the persona draw for a reproducible session"
    )

    # Ollama-specific options
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (Ollama and blind eval, default: 127.0.0.1)"
    )

    # Common options
    parser.add_argument(
        "--port",
        type=int,
        help="Server port (default: 7860 Gradio, 7861 blind eval, 5000 Ollama)"
    )

    args = parser.parse_args()

    # The oracle condition replays real comments, so it needs neither an
    # adapter nor a GPU — it is the one mode that can run before training ends.
    needs_model = not (args.interface == "blind" and args.oracle)

    if needs_model and not args.model:
        parser.error("--model is required (except with --interface blind --oracle)")

    if needs_model and not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("\nMake sure you've trained a model first:")
        print("  python -m litigpt.pipeline --step train")
        sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Set default port if not specified
    if args.port is None:
        args.port = {"gradio": 7860, "blind": 7861}.get(args.interface, 5000)

    # Launch appropriate interface
    if args.interface == "blind":
        launch_blind(
            args.model,
            args.base_model,
            args.config,
            args.share,
            args.port,
            args.host,
            args.oracle,
            args.classifier,
            args.seed,
        )
    elif args.interface == "gradio":
        launch_gradio(
            args.model,
            args.base_model,
            args.config,
            args.share,
            args.port,
        )
    else:
        launch_ollama(
            args.model,
            args.base_model,
            args.config,
            args.host,
            args.port,
        )

if __name__ == "__main__":
    main()
