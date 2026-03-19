"""
Module 7: Complete Pipeline Runner
Orchestrate the entire pipeline from data to deployment
"""

import argparse
import yaml
from pathlib import Path
import sys

from litigpt.data.extraction import RedditDataExtractor
from litigpt.data.preprocessing import RedditDataPreprocessor

class PipelineRunner:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize pipeline with configuration"""
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("=" * 60)
        print("Reddit Chatbot Pipeline")
        print("=" * 60)
    
    def run_data_extraction(self):
        """Step 1: Extract user data from Reddit JSONL files"""

        print("\n[1/5] EXTRACTING DATA")
        print("-" * 60)

        config = self.config['data']
        usernames = config['target_usernames']
        extractor = RedditDataExtractor(config['raw_dir'])

        users_data = extractor.extract_multiple_users(
            usernames=usernames,
            comments_file=config.get("comments_filename", "comments.jsonl"),
            posts_file=config.get("submission_filename", "submissions.jsonl")
        )

        extractor.save_multi_user_data(users_data, config['processed_dir'])

        total = sum(len(d) for d in users_data.values())
        print(f"[OK] Extracted {total} items for {usernames}")
        return users_data
    
    def run_preprocessing(self):
        """Step 2: Preprocess and format data for training"""
        
        print("\n[2/5] PREPROCESSING DATA")
        print("-" * 60)
        
        import polars as pl

        config = self.config['data']
        preprocessor = RedditDataPreprocessor(
            min_length=config['min_comment_length'],
            max_length=config['max_comment_length']
        )

        # Load per-user parquet files saved by run_data_extraction
        processed_dir = Path(config['processed_dir'])
        users_data = {
            p.stem.replace("_data", ""): pl.read_parquet(p)
            for p in sorted(processed_dir.glob("*_data.parquet"))
        }
        if not users_data:
            raise FileNotFoundError(
                f"No *_data.parquet files found in {processed_dir}. Run extraction first."
            )

        # Load all comments for context
        extractor = RedditDataExtractor(config['raw_dir'])
        all_comments = extractor.load_data(config.get("comments_filename", "comments.jsonl"))

        # Filter quality per user
        users_data = {u: preprocessor.filter_quality(d) for u, d in users_data.items()}

        # Create training pairs for all users
        pairs = preprocessor.create_multi_user_training_pairs(users_data, all_comments)

        if len(pairs) < 100:
            print(f"[WARNING]  Warning: Only {len(pairs)} training pairs. Consider using a user with more comments.")

        # Format for training
        formatted = preprocessor.format_for_training(pairs, format_type="chatml")
        
        # Split
        train, val = preprocessor.split_data(formatted, train_ratio=0.9)
        
        # Save
        preprocessor.save_training_data(train, val, config['training_dir'])
        
        print(f"[OK] Created {len(train)} training and {len(val)} validation examples")
        return len(train), len(val)
    
    def run_training(self):
        """Step 3: Fine-tune the model"""
        from litigpt.training.trainer import RedditModelTrainer

        print("\n[3/5] TRAINING MODEL")
        print("-" * 60)

        model_config = self.config['model']
        training_config = self.config['training']
        data_config = self.config['data']
        
        trainer = RedditModelTrainer(
            model_name=model_config['base_model'],
            output_dir=model_config['output_dir']
        )
        
        print(f"Base model: {model_config['base_model']}")
        print(f"Output: {model_config['output_dir']}")
        print(f"Epochs: {training_config['num_epochs']}")
        print(f"Batch size: {training_config['batch_size']}")
        print(f"Learning rate: {training_config['learning_rate']}")
        
        # Train
        trainer.train(
            data_dir=data_config['training_dir'],
            num_epochs=training_config['num_epochs'],
            batch_size=training_config['batch_size'],
            learning_rate=training_config['learning_rate'],
            max_seq_length=training_config['max_seq_length']
        )
        
        print(f"[OK] Model trained and saved to {model_config['output_dir']}")
    
    def run_evaluation(self):
        """Step 4: Test the model interactively"""
        from litigpt.inference.generator import RedditBotInference

        print("\n[4/5] EVALUATING MODEL")
        print("-" * 60)

        model_config = self.config['model']
        inference_config = self.config['inference']
        
        bot = RedditBotInference(
            model_path=model_config['output_dir'],
            base_model=model_config['base_model'],
            use_lora=True,
            load_in_4bit=True
        )
        
        print("Model loaded. Testing with sample contexts...\n")
        
        # Test examples
        test_contexts = [
            "user1: What's your opinion on Python vs JavaScript?",
            "user2: Anyone here play video games? What are you playing?",
            "user3: This subreddit has really grown lately!"
        ]
        
        for i, context in enumerate(test_contexts, 1):
            print(f"Test {i}:")
            print(f"Context: {context}")
            
            response = bot.generate_response(
                context,
                max_new_tokens=inference_config['max_new_tokens'],
                temperature=inference_config['temperature'],
                top_p=inference_config['top_p']
            )
            
            print(f"Response: {response}\n")
            print("-" * 60)
        
        # Interactive mode option
        print("\nWould you like to test interactively? (y/n): ", end="")
        choice = input().strip().lower()
        
        if choice == 'y':
            bot.interactive_mode()
        
        print("[OK] Evaluation complete")
    
    def run_deployment(self):
        """Step 5: Deploy bot to Reddit"""
        from litigpt.deployment.reddit_bot import RedditBot

        print("\n[5/5] DEPLOYING BOT")
        print("-" * 60)

        model_config = self.config['model']
        bot_config = self.config['bot']
        inference_config = self.config['inference']
        
        print(f"Target subreddit: r/{bot_config['subreddit']}")
        print(f"Reply probability: {bot_config['reply_probability']}")
        
        print("\n[WARNING]  IMPORTANT: Make sure you have:")
        print("1. Created a Reddit app at https://www.reddit.com/prefs/apps")
        print("2. Added credentials to .env file")
        print("3. Read the subreddit rules about bots")
        print("4. Consider adding a bot disclaimer to responses")
        
        print("\nReady to deploy? (y/n): ", end="")
        choice = input().strip().lower()
        
        if choice != 'y':
            print("Deployment cancelled.")
            return
        
        # Initialize bot
        bot = RedditBot(
            model_path=model_config['output_dir'],
            base_model=model_config['base_model'],
            subreddit_name=bot_config['subreddit'],
            bot_username=bot_config.get('bot_username', 'your_bot_username'),
            trigger_keywords=bot_config.get('trigger_keywords'),
            reply_probability=bot_config['reply_probability'],
            min_score_threshold=bot_config['min_score_threshold'],
            cooldown_seconds=bot_config['cooldown_seconds']
        )
        
        # Run bot
        print("\n[OK] Bot deployed! Press Ctrl+C to stop.\n")
        bot.run()
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        
        try:
            # Step 1: Extract
            self.run_data_extraction()
            
            # Step 2: Preprocess
            train_size, val_size = self.run_preprocessing()
            
            if train_size < 50:
                print("\n[WARNING]  Warning: Very small training set. Model may not learn effectively.")
                print("Continue anyway? (y/n): ", end="")
                if input().strip().lower() != 'y':
                    return
            
            # Step 3: Train
            self.run_training()
            
            # Step 4: Evaluate
            self.run_evaluation()
            
            # Step 5: Deploy
            print("\nWould you like to deploy the bot now? (y/n): ", end="")
            if input().strip().lower() == 'y':
                self.run_deployment()
            
            print("\n" + "=" * 60)
            print("PIPELINE COMPLETE!")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\nPipeline interrupted by user.")
        except Exception as e:
            print(f"\n[ERROR] Error: {e}")
            import traceback
            traceback.print_exc()

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
