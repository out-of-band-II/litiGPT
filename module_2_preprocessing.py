"""
Module 2: Data Preprocessing
Clean and format data for training
"""

import pandas as pd
import re
from typing import List, Dict, Tuple
import jsonlines
from pathlib import Path

class RedditDataPreprocessor:
    def __init__(self, min_length: int = 10, max_length: int = 512):
        self.min_length = min_length
        self.max_length = max_length
        
    def clean_text(self, text: str) -> str:
        """Clean Reddit text"""
        if not isinstance(text, str):
            return ""
        
        # Remove deleted/removed content
        if text.lower() in ['[deleted]', '[removed]', 'none']:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs (optional - keep if they're part of user's style)
        # text = re.sub(r'http\S+|www.\S+', '[URL]', text)
        
        # Remove Reddit formatting artifacts
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        
        return text.strip()
    
    def filter_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter low-quality entries"""
        
        # Clean body/selftext
        df['body'] = df.apply(
            lambda x: self.clean_text(x.get('body') or x.get('selftext', '')), 
            axis=1
        )
        
        # Filter by length
        df['text_length'] = df['body'].str.len()
        df = df[
            (df['text_length'] >= self.min_length) & 
            (df['text_length'] <= self.max_length)
        ]
        
        # Filter deleted/removed
        df = df[df['body'] != '']
        
        # Filter by score (optional - only keep upvoted content)
        if 'score' in df.columns:
            df = df[df['score'] > 0]
        
        print(f"After filtering: {len(df)} entries")
        return df.reset_index(drop=True)
    
    def create_training_pairs(self, user_data: pd.DataFrame, 
                             all_comments: List[Dict],
                             username: str = None) -> List[Dict]:
        """
        Create (context, response) pairs for training
        
        Args:
            user_data: User's comments/posts
            all_comments: All subreddit comments for context
            username: Username to tag in training data (for multi-user)
        """
        from module_1_data_extraction import RedditDataExtractor
        
        extractor = RedditDataExtractor("data/raw")
        thread_data = extractor.build_conversation_threads(all_comments)
        
        training_pairs = []
        
        for _, row in user_data.iterrows():
            if row['type'] != 'comment':
                continue
            
            # Get context
            context_items = extractor.get_context_for_comment(
                row.to_dict(), 
                thread_data,
                max_context=3
            )
            
            if not context_items:
                continue
            
            # Format context
            context_text = self._format_context(context_items)
            response_text = row['body']
            
            if context_text and response_text:
                pair = {
                    'context': context_text,
                    'response': response_text,
                    'score': row.get('score', 0),
                    'created_utc': row.get('created_utc', 0)
                }
                
                # Add username if doing multi-user training
                if username:
                    pair['username'] = username
                
                training_pairs.append(pair)
        
        print(f"Created {len(training_pairs)} training pairs")
        return training_pairs
    
    def create_multi_user_training_pairs(self, 
                                        users_data: Dict[str, pd.DataFrame],
                                        all_comments: List[Dict]) -> List[Dict]:
        """
        Create training pairs for multiple users
        
        Args:
            users_data: Dictionary mapping username to their data
            all_comments: All subreddit comments
            
        Returns:
            List of training pairs with username tags
        """
        all_pairs = []
        
        for username, user_data in users_data.items():
            print(f"\nProcessing {username}...")
            pairs = self.create_training_pairs(user_data, all_comments, username)
            all_pairs.extend(pairs)
        
        print(f"\nTotal training pairs: {len(all_pairs)}")
        print(f"Users: {list(users_data.keys())}")
        
        return all_pairs
    
    def _format_context(self, context_items: List[Dict]) -> str:
        """Format context items into a single string"""
        formatted = []
        
        for item in context_items:
            author = item.get('author', 'unknown')
            body = self.clean_text(item.get('body') or item.get('selftext', ''))
            
            if body:
                formatted.append(f"{author}: {body}")
        
        return "\n".join(formatted)
    
    def format_for_training(self, pairs: List[Dict], 
                           format_type: str = "chatml",
                           multi_user: bool = False) -> List[Dict]:
        """
        Format training pairs for specific model format
        
        Formats:
        - chatml: ChatML format (for Llama, Mistral)
        - alpaca: Alpaca instruction format
        - raw: Simple context -> response
        
        Args:
            pairs: Training pairs
            format_type: Format to use
            multi_user: Whether this is multi-user training
        """
        
        formatted_data = []
        
        for pair in pairs:
            username = pair.get('username', 'unknown')
            
            if format_type == "chatml":
                # System prompt changes based on multi-user mode
                if multi_user:
                    system_content = f"You are {username}, a Reddit user. Respond in {username}'s writing style and tone."
                else:
                    system_content = "You are a helpful Reddit user responding to comments in a conversational manner."
                
                formatted = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_content
                        },
                        {
                            "role": "user",
                            "content": pair['context']
                        },
                        {
                            "role": "assistant",
                            "content": pair['response']
                        }
                    ]
                }
                
                # Add username metadata for tracking
                if multi_user:
                    formatted['username'] = username
                    
            elif format_type == "alpaca":
                if multi_user:
                    instruction = f"Respond to the following Reddit conversation as {username} would:"
                else:
                    instruction = "Respond to the following Reddit conversation:"
                    
                formatted = {
                    "instruction": instruction,
                    "input": pair['context'],
                    "output": pair['response']
                }
                
                if multi_user:
                    formatted['username'] = username
                    
            else:  # raw
                if multi_user:
                    formatted = {
                        "text": f"### User: {username}\n### Context:\n{pair['context']}\n\n### Response:\n{pair['response']}"
                    }
                else:
                    formatted = {
                        "text": f"### Context:\n{pair['context']}\n\n### Response:\n{pair['response']}"
                    }
            
            formatted_data.append(formatted)
        
        return formatted_data
    
    def split_data(self, data: List[Dict], 
                   train_ratio: float = 0.9) -> Tuple[List[Dict], List[Dict]]:
        """Split into train/validation sets"""
        import random
        random.shuffle(data)
        
        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        print(f"Train: {len(train_data)}, Validation: {len(val_data)}")
        return train_data, val_data
    
    def save_training_data(self, train_data: List[Dict], val_data: List[Dict],
                          output_dir: str = "data/training"):
        """Save formatted training data"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as JSONL
        with jsonlines.open(output_path / "train.jsonl", 'w') as writer:
            writer.write_all(train_data)
        
        with jsonlines.open(output_path / "val.jsonl", 'w') as writer:
            writer.write_all(val_data)
        
        print(f"Saved training data to {output_dir}")

if __name__ == "__main__":
    # Example usage
    preprocessor = RedditDataPreprocessor()
    
    # Load user data
    user_data = pd.read_json("data/processed/user_data.jsonl", lines=True)
    
    # Load all comments for context building
    all_comments = []
    
    from module_1_data_extraction import RedditDataExtractor
    extractor = RedditDataExtractor('data/raw')
    all_comments = extractor._load_reddit_data_file("data/raw/comments.jsonl")
    
    # Filter quality
    user_data = preprocessor.filter_quality(user_data)
    
    # Create training pairs
    pairs = preprocessor.create_training_pairs(user_data, all_comments)
    
    # Format for training
    formatted = preprocessor.format_for_training(pairs, format_type="chatml")
    
    # Split and save
    train, val = preprocessor.split_data(formatted)
    preprocessor.save_training_data(train, val)