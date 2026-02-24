"""
Module 13: User Classification
Automatically detect which user to impersonate based on context
"""

import logging
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class UserClassifier:
    """
    Classify which user's style to use based on conversation context
    """
    
    def __init__(self, users_data: Dict[str, List[str]] = None):
        """
        Initialize classifier
        
        Args:
            users_data: Dictionary mapping username to list of their comments
        """
        self.users_data = users_data or {}
        self.vectorizer = None
        self.user_profiles = {}
        
        if users_data:
            self._build_user_profiles()
    
    def _build_user_profiles(self):
        """Build TF-IDF profiles for each user"""
        
        logger.info("Building user profiles...")
        
        all_texts = []
        user_indices = {}
        current_idx = 0
        
        # Collect all texts and track indices
        for username, texts in self.users_data.items():
            all_texts.extend(texts)
            user_indices[username] = (current_idx, current_idx + len(texts))
            current_idx += len(texts)
        
        # Fit vectorizer on all texts
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        all_vectors = self.vectorizer.fit_transform(all_texts)
        
        # Create average profile for each user
        for username, (start_idx, end_idx) in user_indices.items():
            user_vectors = all_vectors[start_idx:end_idx]
            avg_vector = user_vectors.mean(axis=0)
            self.user_profiles[username] = avg_vector
            
            logger.info(f"  {username}: {end_idx - start_idx} samples")

        logger.info(f"User profiles built for {len(self.user_profiles)} users")
    
    def classify_context(self, context: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Classify which user's style best matches the context
        
        Args:
            context: The conversation context
            top_k: Return top K most similar users
            
        Returns:
            List of (username, similarity_score) tuples
        """
        
        if not self.vectorizer or not self.user_profiles:
            raise ValueError("User profiles not built. Call _build_user_profiles() first.")
        
        # Vectorize context
        context_vector = self.vectorizer.transform([context])
        
        # Calculate similarity with each user profile
        similarities = {}
        for username, profile in self.user_profiles.items():
            sim = cosine_similarity(context_vector, profile)[0][0]
            similarities[username] = sim
        
        # Sort by similarity
        ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]
    
    def predict_user(self, context: str, threshold: float = 0.1) -> str:
        """
        Predict single best user for context
        
        Args:
            context: Conversation context
            threshold: Minimum similarity threshold
            
        Returns:
            Username or None if no good match
        """
        
        results = self.classify_context(context, top_k=1)
        
        if not results:
            return None
        
        username, score = results[0]
        
        if score < threshold:
            return None
        
        return username
    
    def save_profiles(self, output_path: str):
        """Save user profiles for later use"""
        
        import pickle
        
        data = {
            'vectorizer': self.vectorizer,
            'user_profiles': self.user_profiles,
            'users': list(self.user_profiles.keys())
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"User profiles saved to {output_path}")
    
    def load_profiles(self, input_path: str):
        """Load pre-built user profiles"""
        
        import pickle
        
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        self.vectorizer = data['vectorizer']
        self.user_profiles = data['user_profiles']
        
        logger.info(f"Loaded profiles for {len(self.user_profiles)} users: {data['users']}")

class KeywordUserSelector:
    """
    Simple keyword-based user selection
    Useful for topic-based routing
    """
    
    def __init__(self, user_keywords: Dict[str, List[str]]):
        """
        Args:
            user_keywords: Dict mapping username to list of keywords they discuss
            
        Example:
            {
                'tech_user': ['python', 'javascript', 'coding', 'programming'],
                'gaming_user': ['valorant', 'league', 'gaming', 'fps'],
                'fitness_user': ['gym', 'workout', 'protein', 'running']
            }
        """
        self.user_keywords = {
            username: [kw.lower() for kw in keywords]
            for username, keywords in user_keywords.items()
        }
    
    def predict_user(self, context: str) -> str:
        """
        Select user based on keyword matching
        
        Args:
            context: Conversation context
            
        Returns:
            Username with most keyword matches
        """
        
        context_lower = context.lower()
        
        scores = {}
        for username, keywords in self.user_keywords.items():
            score = sum(1 for kw in keywords if kw in context_lower)
            scores[username] = score
        
        if not scores or max(scores.values()) == 0:
            return None
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def get_matching_keywords(self, context: str, username: str) -> List[str]:
        """Get which keywords matched for a user"""
        
        context_lower = context.lower()
        keywords = self.user_keywords.get(username, [])
        
        return [kw for kw in keywords if kw in context_lower]

class HybridUserSelector:
    """
    Combine multiple selection strategies
    """
    
    def __init__(self, 
                 tfidf_classifier: UserClassifier = None,
                 keyword_selector: KeywordUserSelector = None,
                 default_user: str = None):
        """
        Args:
            tfidf_classifier: TF-IDF based classifier
            keyword_selector: Keyword based selector
            default_user: Fallback user if no good match
        """
        self.tfidf_classifier = tfidf_classifier
        self.keyword_selector = keyword_selector
        self.default_user = default_user
    
    def predict_user(self, context: str, method: str = 'auto') -> Dict:
        """
        Predict user with multiple strategies
        
        Args:
            context: Conversation context
            method: 'auto', 'tfidf', 'keyword', or 'hybrid'
            
        Returns:
            Dict with prediction details
        """
        
        result = {
            'username': None,
            'method': method,
            'confidence': 0.0,
            'alternatives': []
        }
        
        if method in ['auto', 'hybrid', 'tfidf']:
            if self.tfidf_classifier:
                tfidf_results = self.tfidf_classifier.classify_context(context, top_k=3)
                result['tfidf_top'] = tfidf_results
                
                if tfidf_results:
                    username, score = tfidf_results[0]
                    result['username'] = username
                    result['confidence'] = score
                    result['alternatives'] = [u for u, s in tfidf_results[1:]]
        
        if method in ['auto', 'hybrid', 'keyword']:
            if self.keyword_selector:
                keyword_user = self.keyword_selector.predict_user(context)
                result['keyword_user'] = keyword_user
                
                # If keyword method has a strong opinion, use it
                if keyword_user and method == 'keyword':
                    result['username'] = keyword_user
                    result['method'] = 'keyword'
        
        # Hybrid: Use keyword if it matches, otherwise TF-IDF
        if method == 'hybrid' and result.get('keyword_user'):
            result['username'] = result['keyword_user']
            result['method'] = 'hybrid'
        
        # Fallback to default
        if not result['username'] and self.default_user:
            result['username'] = self.default_user
            result['method'] = 'default'
        
        return result

def build_user_classifier_from_data(processed_dir: str = "data/processed") -> UserClassifier:
    """
    Build classifier from processed user data
    
    Args:
        processed_dir: Directory with user data files
        
    Returns:
        Trained UserClassifier
    """
    
    import polars as pl

    users_data = {}
    processed_path = Path(processed_dir)

    # Load metadata to get user list
    metadata_file = processed_path / "users_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"No metadata found at {metadata_file}")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Load each user's comments
    for username in metadata['users']:
        # Support both parquet and jsonl formats
        user_parquet = processed_path / f"{username}_data.parquet"
        user_jsonl = processed_path / f"{username}_data.jsonl"

        if user_parquet.exists():
            df = pl.read_parquet(user_parquet)
        elif user_jsonl.exists():
            df = pl.read_ndjson(user_jsonl, infer_schema_length=None)
        else:
            logger.warning(f"No data file found for {username}")
            continue

        # Extract text content
        texts = []
        for row in df.to_dicts():
            text = row.get('body') or row.get('selftext', '')
            if text and text not in ['[deleted]', '[removed]']:
                texts.append(text)

        users_data[username] = texts
        logger.info(f"Loaded {len(texts)} texts for {username}")
    
    # Build classifier
    classifier = UserClassifier(users_data)
    
    return classifier

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example 1: TF-IDF based classification
    print("="*60)
    print("Example 1: TF-IDF Classification")
    print("="*60)
    
    # Build from processed data
    classifier = build_user_classifier_from_data("data/processed")
    
    # Save for later use
    classifier.save_profiles("models/user_classifier.pkl")
    
    # Test classification
    test_context = "I love Python and machine learning! Been working on a new neural network."
    
    results = classifier.classify_context(test_context, top_k=3)
    print(f"\nContext: {test_context}")
    print("\nTop matches:")
    for username, score in results:
        print(f"  {username}: {score:.3f}")
    
    predicted = classifier.predict_user(test_context)
    print(f"\nPredicted user: {predicted}")
    
    # Example 2: Keyword based selection
    print("\n" + "="*60)
    print("Example 2: Keyword Selection")
    print("="*60)
    
    keyword_selector = KeywordUserSelector({
        'tech_enthusiast': ['python', 'javascript', 'coding', 'api', 'github'],
        'gamer': ['valorant', 'league', 'fps', 'gaming', 'rank'],
        'fitness_guru': ['gym', 'workout', 'protein', 'cardio', 'gains']
    })
    
    test_contexts = [
        "Just hit a new PR at the gym! 225 bench press!",
        "What's the best Python framework for web development?",
        "Anyone else stuck in silver rank in Valorant?"
    ]
    
    for ctx in test_contexts:
        user = keyword_selector.predict_user(ctx)
        keywords = keyword_selector.get_matching_keywords(ctx, user) if user else []
        print(f"\nContext: {ctx}")
        print(f"Predicted: {user}")
        print(f"Matched keywords: {keywords}")
    
    # Example 3: Hybrid approach
    print("\n" + "="*60)
    print("Example 3: Hybrid Selection")
    print("="*60)
    
    hybrid = HybridUserSelector(
        tfidf_classifier=classifier,
        keyword_selector=keyword_selector,
        default_user='general_user'
    )
    
    result = hybrid.predict_user(test_context, method='hybrid')
    print(f"\nContext: {test_context}")
    print(f"Prediction: {json.dumps(result, indent=2)}")
