"""
Module 1: Data Extraction
Extract target user's comments and posts from Reddit JSONL data
"""

import jsonlines
import pandas as pd
from pathlib import Path
from typing import Dict, List
import json
import zstandard as zstd
import io

class RedditDataExtractor:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load_jsonl(self, filename: str) -> List[Dict]:
        """Load a JSONL file"""
        data = []
        filepath = self.data_dir / filename
        with jsonlines.open(filepath) as reader:
            for obj in reader:
                data.append(obj)
        return data
    
    def extract_zstd(self,filename,condition=None):
        """
        Processes a JSON stream from a compressed file and yields objects based on condition.

        Args:
            filepath (str): Path to the compressed JSON file
            condition (callable): Function to evaluate each object; if None, all objects are yielded

        Yields:
            dict: Each JSON object that meets the condition
        """
        i=0
        filepath = self.data_dir /filename
        with open(filepath, 'rb') as compressed_file:
            dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
            with dctx.stream_reader(compressed_file) as stream_reader:
                # Read all content into a buffer
                text_content = io.TextIOWrapper(stream_reader, encoding='utf-8')
                for line in text_content:
                    obj = json.loads(line)
                    if condition is None or condition(obj):
                        i=i+1
                        if i%1000==0:
                            print (i, ' comments collected.')
                        yield obj

    def _load_reddit_data_file(self,filename:str):
        admitted_ext = [".jsonl",".zst",".parquet"]
        print(f"Loading comments from {filename}...")
        if Path(filename).suffix == ".jsonl":
            data = self.load_jsonl(filename)
        elif Path(filename).suffix == ".zst":
            data :List[Dict] = [entry for entry in self.extract_zstd(filename)]
        elif Path(filename).suffix == ".parquet":
            data = pd.read_parquet(self.data_dir/filename).to_dict('records')
        else:
            raise ValueError(f"{filename} suffix ({Path(filename).suffix}) not recognized (must be onf of {', '.join(admitted_ext)})")
        
        return data
        
    def extract_user_data(self, username: str, 
                         comments_file: str = "comments.jsonl",
                         posts_file: str = "submissions.jsonl") -> pd.DataFrame:
        """Extract all comments and posts from a specific user"""
        
        # Load comments
        print(f"Loading comments from {posts_file}...")
        comments = self._load_reddit_data_file(comments_file)
        user_comments = [c for c in comments if c.get('author') == username]
        
        # Load posts
        print(f"Loading posts from {posts_file}...")
        posts = self._load_reddit_data_file(posts_file)
        user_posts = [p for p in posts if p.get('author') == username]
        
        # Convert to DataFrame
        comments_df = pd.DataFrame(user_comments)
        posts_df = pd.DataFrame(user_posts)
        
        # Add type column
        if not comments_df.empty:
            comments_df['type'] = 'comment'
        if not posts_df.empty:
            posts_df['type'] = 'post'
        
        # Combine
        user_data = pd.concat([comments_df, posts_df], ignore_index=True)
        
        print(f"Extracted {len(user_comments)} comments and {len(user_posts)} posts")
        return user_data
    
    def build_conversation_threads(self, all_data: List[Dict]) -> Dict[str, List[Dict]]:
        """Build conversation threads from all subreddit data"""
        
        # Create lookup dictionaries
        comments_by_id = {}
        comments_by_parent = {}
        posts_by_id = {}
        
        for item in all_data:
            item_id = item.get('id') or item.get('name', '').split('_')[-1]
            
            if item.get('type') == 'comment' or 'parent_id' in item:
                comments_by_id[item_id] = item
                parent_id = item.get('parent_id', '').split('_')[-1]
                if parent_id not in comments_by_parent:
                    comments_by_parent[parent_id] = []
                comments_by_parent[parent_id].append(item)
            else:
                posts_by_id[item_id] = item
        
        return {
            'comments_by_id': comments_by_id,
            'comments_by_parent': comments_by_parent,
            'posts_by_id': posts_by_id
        }
    
    def get_context_for_comment(self, comment: Dict, thread_data: Dict, 
                                max_context: int = 5) -> List[Dict]:
        """Get parent comments/posts for context"""
        context = []
        current = comment
        
        for _ in range(max_context):
            parent_id = current.get('parent_id', '').split('_')[-1]
            if not parent_id:
                break
            
            # Check if parent is a comment
            parent = thread_data['comments_by_id'].get(parent_id)
            if parent:
                context.insert(0, parent)
                current = parent
            else:
                # Check if parent is a post
                parent = thread_data['posts_by_id'].get(parent_id)
                if parent:
                    context.insert(0, parent)
                break
        
        return context
    
    def save_processed_data(self, data: pd.DataFrame, output_path: str):
        """Save processed data"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_json(output_path, orient='records', lines=True)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    extractor = RedditDataExtractor("data/raw")
    
    # Extract specific user's data
    user_data = extractor.extract_user_data("target_username")
    
    # Save
    extractor.save_processed_data(user_data, "data/processed/user_data.jsonl")
