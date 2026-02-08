"""
Module 1: Data Extraction (Polars Version)
"""

import polars as pl
import jsonlines
from pathlib import Path
from typing import Dict, List, Union
from argparse import ArgumentParser

class RedditDataExtractor:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load_data(self, filename: str) -> pl.DataFrame:
        """
        Carica file JSONL o Parquet automaticamente
        """
        filepath = self.data_dir / filename
        
        if filepath.suffix == '.parquet':
            return pl.read_parquet(filepath)
        elif filepath.suffix == '.jsonl':
            return pl.read_ndjson(filepath,infer_schema_length=None)
        else:
            raise ValueError(f"Formato non supportato: {filepath.suffix}")
    
    def extract_user_data(self, 
                         username: str,
                         comments_file: str = "comments.jsonl",
                         posts_file: str = "submissions.jsonl") -> pl.DataFrame:
        """
        Estrae dati utente (versione Polars)
        """
        print(f"Caricamento commenti da {comments_file}...")
        comments = self.load_data(comments_file)
        
        # Filtra per utente
        user_comments = comments.filter(pl.col('author') == username)
        user_comments = user_comments.with_columns(pl.lit('comment').alias('type'))
        print(user_comments.head())
        
        print(f"Caricamento post da {posts_file}...")
        posts = self.load_data(posts_file)
        
        user_posts = posts.filter(pl.col('author') == username)
        user_posts = user_posts.with_columns(pl.lit('post').alias('type'))
        
        # Combina (solo colonne comuni)
        user_data = pl.concat([
            user_comments,
            user_posts
        ],how='diagonal')
        
        print(f"Estratti {len(user_comments)} commenti e {len(user_posts)} post")
        return user_data
    
    def extract_multiple_users(self,
                               usernames: List[str],
                               comments_file: str = "comments.jsonl",
                               posts_file: str = "submissions.jsonl",
                               min_comments_per_user: int = 100) -> Dict[str, pl.DataFrame]:
        """
        Estrae dati per utenti multipli (versione Polars)
        """
        users_data = {}
        
        # Carica una sola volta
        # print("Caricamento dati completi...")
        # comments = self.load_data(comments_file)
        # posts = self.load_data(posts_file)
        
        for username in usernames:
            print(f"\n{'='*60}")
            print(f"Processing user: {username}")
            print('='*60)
            
            # Filtra per utente
            user_data = self.extract_user_data(username,comments_file,posts_file)
            
            users_data[username] = user_data
            print(f"✓ Added {username} with {len(user_data)} items")
        
        print(f"\n{'='*60}")
        print(f"Total users extracted: {len(users_data)}")
        print('='*60)
        
        return users_data
    
    def save_processed_data(self, data: pl.DataFrame, output_path: str):
        """
        Salva dati processati come JSONL o Parquet
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if output_file.suffix == '.parquet':
            data.write_parquet(output_path, compression='snappy')
        elif output_file.suffix == '.jsonl':
            data.write_ndjson(output_path)
        else:
            # Default: JSONL per compatibilità
            data.write_ndjson(output_path)
        
        print(f"Salvati dati in {output_path}")
    
    def save_multi_user_data(self, 
                            users_data: Dict[str, pl.DataFrame],
                            output_dir: str = "data/processed"):
        """
        Salva dati multi-utente
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Salva ogni utente
        for username, data in users_data.items():
            user_file = output_path / f"{username}_data.parquet"
            data.write_parquet(user_file, compression='snappy')
            print(f"Saved {username}: {user_file}")
        
        # Metadata
        import json
        metadata = {
            'users': list(users_data.keys()),
            'user_stats': {
                username: {
                    'total_items': len(data),
                    'comments': len(data.filter(pl.col('type') == 'comment')),
                    'posts': len(data.filter(pl.col('type') == 'post'))
                }
                for username, data in users_data.items()
            }
        }
        
        metadata_file = output_path / "users_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nMetadata saved: {metadata_file}")
    # OPTIMIZE THIS

    # def build_conversation_threads(self, all_data: pl.DataFrame) -> Dict:
    #     """
    #     Costruisce thread conversazioni (versione Polars)
    #     """
    #     # Converti a dizionari per lookup veloce
    #     comments = all_data.filter(
    #          pl.col('parent_id').is_not_null() #(pl.col('type') == 'comment') | need to find better way
    #     )
        
    #     comments_by_id = {
    #         row['id']: row 
    #         for row in comments.to_dicts()
    #     }
        
    #     comments_by_parent = {}
    #     for row in comments.to_dicts():
    #         parent_id = row.get('parent_id', '').split('_')[-1]
    #         if parent_id not in comments_by_parent:
    #             comments_by_parent[parent_id] = []
    #         comments_by_parent[parent_id].append(row)
        
    #     posts = all_data.filter(pl.col('type') == 'post')
    #     posts_by_id = {
    #         row['id']: row 
    #         for row in posts.to_dicts()
    #     }
        
    #     return {
    #         'comments_by_id': comments_by_id,
    #         'comments_by_parent': comments_by_parent,
    #         'posts_by_id': posts_by_id
    #     }
    
    def build_conversation_threads(self, all_data_pl: pl.DataFrame) -> Dict[str, List[Dict]]:
            """Build conversation threads from all subreddit data"""
            
            all_data = all_data_pl.to_dicts()
            # Create lookup dictionaries
            comments_by_id = {}
            comments_by_parent = {}
            posts_by_id = {}
            
            for item in all_data:
                item_id = item.get('id') or item.get('name', '').split('_')[-1]
                
                if item.get('type') == 'comment' or 'parent_id' in item:
                    comments_by_id[item_id] = item
                    pid = item.get('parent_id', '')
                    try:
                        parent_id = pid.split('_')[-1]
                    except:
                        print(f"Error processign {item}")
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
    
    # def save_processed_data(self, data: pl.DataFrame, output_path: str):
    #     """Save processed data"""
    #     Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    #     data.to_json(output_path, orient='records', lines=True)
    #     print(f"Saved to {output_path}")

def data_extraction_parser():
    parser = ArgumentParser(description="Data extraction module",
                            epilog=f"""
    Examples:
    python %(prog)s -c litigi_comments.parquet -s submissions.jsonl --multi-user
    """)

    parser.add_argument("--dir","-d", required=False, default="data/raw", help="Data directory")
    parser.add_argument("--comments","-c", required=False, default="comments.parquet",
                       help="Comment data")
    parser.add_argument("--submissions","-s", required=False, default="submissions.jsonl", help="Submissison data")
    parser.add_argument("--multi-user", action="store_true", 
                       help="Enable multi-user mode")
    parser.add_argument("--users", nargs="+", type= str,
                       help="users to extract")

    return parser
    

if __name__ == "__main__":
    # Example usage - Single user
    parser = data_extraction_parser()
    args = parser.parse_args()
    submission_data_file = args.submissions
    comments_data_file = args.comments
    users = args.users

    multi_user:bool = args.multi_user
    extractor = RedditDataExtractor("data/raw")
    if not multi_user:
        target_username = users[0]
        if len(users) > 1:
            import warnings
            warnings.warn(f"No multi user option selected, processing only  {target_username}")
        
        user_data = extractor.extract_user_data(target_username,comments_data_file,submission_data_file)
        extractor.save_processed_data(user_data, "data/processed/user_data.jsonl")
    else:
        # Example usage - Multiple users
        users_data = extractor.extract_multiple_users(
            usernames=users,
            min_comments_per_user=100,
            comments_file=comments_data_file,
            posts_file = submission_data_file
        )
        extractor.save_multi_user_data(users_data, "data/processed")