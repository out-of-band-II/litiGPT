"""
Module 7: Complete Pipeline Runner
Orchestrate the entire pipeline from data to deployment
"""

import argparse
import logging
from pathlib import Path
import sys

from litigpt.config import Config
from litigpt.data.extraction import RedditDataExtractor
from litigpt.data.preprocessing import RedditDataPreprocessor

logger = logging.getLogger(__name__)

class PipelineRunner:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize pipeline with configuration"""
        self.config = Config.from_yaml(config_path)
        logger.info("=" * 60)
        logger.info("Reddit Chatbot Pipeline")
        logger.info("=" * 60)

    def run_data_extraction(self):
        """Step 1: Extract user data from Reddit JSONL files"""

        logger.info("[1/5] EXTRACTING DATA")
        logger.info("-" * 60)

        data = self.config.data
        extractor = RedditDataExtractor(data.raw_dir)

        users_data = extractor.extract_multiple_users(
            usernames=data.target_usernames,
            comments_file=data.comments_filename,
            posts_file=data.submission_filename,
        )

        extractor.save_multi_user_data(users_data, data.processed_dir)

        total = sum(len(d) for d in users_data.values())
        logger.info("Extracted %d items for %s", total, data.target_usernames)
        return users_data

    def run_preprocessing(self):
        """Step 2: Preprocess and format data for training"""

        logger.info("[2/5] PREPROCESSING DATA")
        logger.info("-" * 60)

        import polars as pl

        data = self.config.data
        preprocessor = RedditDataPreprocessor(
            min_length=data.min_comment_length,
            max_length=data.max_comment_length,
        )

        # Load per-user parquet files saved by run_data_extraction
        processed_dir = Path(data.processed_dir)
        users_data = {
            p.stem.replace("_data", ""): pl.read_parquet(p)
            for p in sorted(processed_dir.glob("*_data.parquet"))
        }
        if not users_data:
            raise FileNotFoundError(
                f"No *_data.parquet files found in {processed_dir}. Run extraction first."
            )

        # Load all comments for context
        extractor = RedditDataExtractor(data.raw_dir)
        all_comments = extractor.load_data(data.comments_filename)

        # Filter quality per user
        users_data = {u: preprocessor.filter_quality(d) for u, d in users_data.items()}

        # Create training pairs for all users
        pairs = preprocessor.create_multi_user_training_pairs(users_data, all_comments)

        if len(pairs) < 100:
            logger.warning("Only %d training pairs. Consider using a user with more comments.", len(pairs))

        # Format for training
        formatted = preprocessor.format_for_training(pairs, format_type="chatml")

        # Split
        train, val = preprocessor.split_data(
            formatted, train_ratio=self.config.training.train_ratio
        )

        # Save
        preprocessor.save_training_data(train, val, data.training_dir)

        logger.info("Created %d training and %d validation examples", len(train), len(val))
        return len(train), len(val)

    def run_training(self):
        """Step 3: Fine-tune the model"""
        from litigpt.training.trainer import RedditModelTrainer
        from litigpt.training.tracking import MLflowTracker
        import mlflow
        import torch
        import jsonlines
        import os
        from collections import Counter

        logger.info("[3/5] TRAINING MODEL")
        logger.info("-" * 60)

        model_cfg = self.config.model
        training_cfg = self.config.training
        data_cfg = self.config.data

        # Initialize MLflow
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        tracker = MLflowTracker(
            experiment_name="reddit-chatbot-training",
            tracking_uri=tracking_uri,
        )

        model_short = model_cfg.base_model.split("/")[-1]
        run_name = f"train_{'_'.join(data_cfg.target_usernames)}_{model_short}"
        tracker.start_run(
            run_name=run_name,
            tags={
                "model": model_cfg.base_model,
                "users": ",".join(data_cfg.target_usernames),
            },
        )

        try:
            # Log full config as params
            tracker.log_config(self.config.model_dump())

            # Log config.yaml as artifact for exact reproducibility
            if Path("config.yaml").exists():
                mlflow.log_artifact("config.yaml")

            # Log dataset sizes and data quality metrics
            train_path = Path(data_cfg.training_dir) / "train.jsonl"
            val_path = Path(data_cfg.training_dir) / "val.jsonl"
            if train_path.exists() and val_path.exists():
                with jsonlines.open(train_path) as r:
                    train_examples = list(r)
                train_size = len(train_examples)
                with jsonlines.open(val_path) as r:
                    val_size = sum(1 for _ in r)
                mlflow.log_metric("train_size", train_size)
                mlflow.log_metric("val_size", val_size)

                # Data quality: response lengths and per-user sample counts
                response_lengths = []
                user_counts = Counter()
                for ex in train_examples:
                    msgs = ex.get("messages", [])
                    if msgs:
                        response_lengths.append(len(msgs[-1].get("content", "")))
                    user_counts[ex.get("username", "unknown")] += 1
                if response_lengths:
                    mlflow.log_metric("avg_response_length", sum(response_lengths) / len(response_lengths))
                for user, count in user_counts.items():
                    mlflow.log_metric(f"user_{user}_samples", count)
                mlflow.log_metric("effective_batch_size",
                                  training_cfg.batch_size * training_cfg.gradient_accumulation_steps)

            # Hardware and environment info
            mlflow.log_param("pytorch_version", torch.__version__)
            mlflow.log_param("cuda_available", torch.cuda.is_available())
            if torch.cuda.is_available():
                mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
                mlflow.log_param("cuda_version", torch.version.cuda)

            # Resolve report_to: collect all available backends
            backends = []
            if mlflow.active_run() is not None:
                backends.append("mlflow")
            try:
                import tensorboard  # noqa: F401
                backends.append("tensorboard")
            except ImportError:
                pass
            report_to = backends if backends else "none"
            logger.info("Reporting to: %s", report_to)
            logger.info("Base model: %s", model_cfg.base_model)
            logger.info("Output: %s", model_cfg.output_dir)
            logger.info("Epochs: %d", training_cfg.num_epochs)
            logger.info("Batch size: %d", training_cfg.batch_size)
            logger.info("Learning rate: %s", training_cfg.learning_rate)

            lora_cfg = self.config.lora

            trainer = RedditModelTrainer(
                model_name=model_cfg.base_model,
                output_dir=model_cfg.output_dir,
            )

            trainer.train(
                data_dir=data_cfg.training_dir,
                num_epochs=training_cfg.num_epochs,
                batch_size=training_cfg.batch_size,
                learning_rate=training_cfg.learning_rate,
                max_seq_length=training_cfg.max_seq_length,
                gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
                lora_r=lora_cfg.r,
                lora_alpha=lora_cfg.lora_alpha,
                lora_dropout=lora_cfg.lora_dropout,
                lora_target_modules=lora_cfg.target_modules,
                report_to=report_to,
            )

            # Log LoRA adapter config (small file, captures adapter architecture)
            adapter_config = Path(model_cfg.output_dir) / "adapter_config.json"
            if adapter_config.exists():
                mlflow.log_artifact(str(adapter_config), artifact_path="model")

            mlflow.set_tag("model_local_path", model_cfg.output_dir)
            logger.info("Model trained and saved to %s", model_cfg.output_dir)

        finally:
            tracker.end_run()

    def run_evaluation(self):
        """Step 4: Test the model interactively"""
        from litigpt.inference.generator import RedditBotInference
        from litigpt.training.tracking import MLflowTracker
        import mlflow
        import os

        logger.info("[4/5] EVALUATING MODEL")
        logger.info("-" * 60)

        model_cfg = self.config.model
        inference_cfg = self.config.inference

        bot = RedditBotInference(
            model_path=model_cfg.output_dir,
            base_model=model_cfg.base_model,
            use_lora=True,
            load_in_4bit=True,
        )

        logger.info("Model loaded. Testing with sample contexts...")

        # Test examples
        test_contexts = [
            "user1: What's your opinion on Python vs JavaScript?",
            "user2: Anyone here play video games? What are you playing?",
            "user3: This subreddit has really grown lately!",
        ]

        samples = []
        for i, context in enumerate(test_contexts, 1):
            print(f"Test {i}:")
            print(f"Context: {context}")

            response = bot.generate_response(
                context,
                max_new_tokens=inference_cfg.max_new_tokens,
                temperature=inference_cfg.temperature,
                top_p=inference_cfg.top_p,
            )

            print(f"Response: {response}\n")
            print("-" * 60)
            samples.append({"context": context, "response": response})

        # Log sample outputs to MLflow if tracking is available
        try:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
            tracker = MLflowTracker(
                experiment_name="reddit-chatbot-training",
                tracking_uri=tracking_uri,
            )
            tracker.start_run(run_name="evaluation_samples")
            tracker.log_sample_outputs(samples)
            tracker.end_run()
        except Exception:
            pass  # MLflow is optional for evaluation

        # Interactive mode option
        print("\nWould you like to test interactively? (y/n): ", end="")
        choice = input().strip().lower()

        if choice == "y":
            bot.interactive_mode()

        logger.info("Evaluation complete")

    def run_deployment(self):
        """Step 5: Deploy bot to Reddit"""
        from litigpt.deployment.reddit_bot import RedditBot

        logger.info("[5/5] DEPLOYING BOT")
        logger.info("-" * 60)

        model_cfg = self.config.model
        bot_cfg = self.config.bot

        logger.info("Target subreddit: r/%s", bot_cfg.subreddit)
        logger.info("Reply probability: %s", bot_cfg.reply_probability)

        logger.warning(
            "Make sure you have: 1) Created a Reddit app, "
            "2) Added credentials to .env, "
            "3) Read subreddit rules about bots, "
            "4) Added a bot disclaimer to responses"
        )

        print("\nReady to deploy? (y/n): ", end="")
        choice = input().strip().lower()

        if choice != "y":
            logger.info("Deployment cancelled.")
            return

        # Initialize bot
        bot = RedditBot(
            model_path=model_cfg.output_dir,
            base_model=model_cfg.base_model,
            subreddit_name=bot_cfg.subreddit,
            bot_username=bot_cfg.bot_username,
            trigger_keywords=bot_cfg.trigger_keywords or None,
            reply_probability=bot_cfg.reply_probability,
            min_score_threshold=bot_cfg.min_score_threshold,
            cooldown_seconds=bot_cfg.cooldown_seconds,
            available_users=bot_cfg.available_users or None,
            user_classifier_path=bot_cfg.user_classifier_path,
            inference_config=self.config.inference.model_dump(),
        )

        # Run bot
        logger.info("Bot deployed! Press Ctrl+C to stop.")
        bot.run()

    def run_full_pipeline(self):
        """Run the complete pipeline"""

        try:
            # Step 1: Extract
            self.run_data_extraction()

            # Step 2: Preprocess
            train_size, val_size = self.run_preprocessing()

            if train_size < 50:
                logger.warning("Very small training set (%d). Model may not learn effectively.", train_size)
                print("Continue anyway? (y/n): ", end="")
                if input().strip().lower() != "y":
                    return

            # Step 3: Train
            self.run_training()

            # Step 4: Evaluate
            self.run_evaluation()

            # Step 5: Deploy
            print("\nWould you like to deploy the bot now? (y/n): ", end="")
            if input().strip().lower() == "y":
                self.run_deployment()

            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETE!")
            logger.info("=" * 60)

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user.")
        except Exception as e:
            logger.error("Error: %s", e, exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="Reddit Chatbot Pipeline")
    parser.add_argument(
        '--step',
        choices=['extract', 'preprocess', 'train', 'eval', 'deploy', 'all'],
        default='all',
        help='Pipeline step to run'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = PipelineRunner(args.config)

    # Run requested step
    if args.step == 'extract':
        pipeline.run_data_extraction()
    elif args.step == 'preprocess':
        pipeline.run_preprocessing()
    elif args.step == 'train':
        pipeline.run_training()
    elif args.step == 'eval':
        pipeline.run_evaluation()
    elif args.step == 'deploy':
        pipeline.run_deployment()
    else:  # all
        pipeline.run_full_pipeline()

if __name__ == "__main__":
    main()
