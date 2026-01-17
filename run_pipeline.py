"""
Module 7: Complete Pipeline Runner
Orchestrate the entire pipeline from data to deployment
"""

import argparse
import yaml
from pathlib import Path
import sys

# Import all modules
from module_1_data_extraction import RedditDataExtractor
from module_2_preprocessing import RedditDataPreprocessor
from module_3_training import RedditModelTrainer
from module_4_inference import RedditBotInference
from module_5_deployment import RedditBot

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
        extractor = RedditDataExtractor(config['raw_dir'])
        
        # Extract user data
        user_data = extractor.extract_user_data(
            username=config['target_username'],
            comments_file="comments.jsonl",
            posts_file="submissions.jsonl"
        )
        
        # Save
        output_path = f"{config['processed_dir']}/user_data.jsonl"
        extractor.save_processed_data(user_data, output_path)
        
        print(f"✓ Extracted {len(user_data)} items from {config['target_username']}")
        return user_data
    
    def run_preprocessing(self):
        """Step 2: Preprocess and format data for training"""
        
        print("\n[2/5] PREPROCESSING DATA")
        print("-" * 60)
        
        import pandas as pd
        import jsonlines
        
        config = self.config['data']
        preprocessor = RedditDataPreprocessor(
            min_length=config['min_comment_length'],
            max_length=config['max_comment_length']
        )
        
        # Load user data
        user_data = pd.read_json(
            f"{config['processed_dir']}/user_data.jsonl",
            lines=True
        )
        
        # Load all comments for context
        all_comments = []
        comments_path = f"{config['raw_dir']}/comments.jsonl"
        with jsonlines.open(comments_path) as reader:
            all_comments = list(reader)
        
        # Filter quality
        user_data = preprocessor.filter_quality(user_data)
        
        # Create training pairs
        pairs = preprocessor.create_training_pairs(user_data, all_comments)
        
        if len(pairs) < 100:
            print(f"⚠️  Warning: Only {len(pairs)} training pairs. Consider using a user with more comments.")
        
        # Format for training
        formatted = preprocessor.format_for_training(pairs, format_type="chatml")
        
        # Split
        train, val = preprocessor.split_data(formatted, train_ratio=0.9)
        
        # Save
        preprocessor.save_training_data(train, val, config['training_dir'])
        
        print(f"✓ Created {len(train)} training and {len(val)} validation examples")
        return len(train), len(val)
    
    def run_training(self):
        """Step 3: Fine-tune the model"""
        
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
        
        print(f"✓ Model trained and saved to {model_config['output_dir']}")
    
    def run_evaluation(self):
        """Step 4: Test the model interactively"""
        
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
        
        print("✓ Evaluation complete")
    
    def run_deployment(self):
        """Step 5: Deploy bot to Reddit"""
        
        print("\n[5/5] DEPLOYING BOT")
        print("-" * 60)
        
        model_config = self.config['model']
        bot_config = self.config['bot']
        inference_config = self.config['inference']
        
        print(f"Target subreddit: r/{bot_config['subreddit']}")
        print(f"Reply probability: {bot_config['reply_probability']}")
        
        print("\n⚠️  IMPORTANT: Make sure you have:")
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
        print("\n✓ Bot deployed! Press Ctrl+C to stop.\n")
        bot.run()
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        
        try:
            # Step 1: Extract
            self.run_data_extraction()
            
            # Step 2: Preprocess
            train_size, val_size = self.run_preprocessing()
            
            if train_size < 50:
                print("\n⚠️  Warning: Very small training set. Model may not learn effectively.")
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
            print(f"\n❌ Error: {e}")
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
