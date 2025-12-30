import pickle
import os
import numpy as np
from usearch.index import Index
from thefuzz import process

class TVShowRecommender:
    def __init__(self, pickle_path="show_embeddings.pkl"):
        """
        Initialize: Load data, map titles to IDs, and build the O(1) Search Index.
        """
        if not os.path.exists(pickle_path):
             raise FileNotFoundError(f"Pickle file not found at {pickle_path}. Please run preprocess_embeddings.py first.")
             
        print(f"Loading data from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            self.show_data = pickle.load(f)
            
        self.all_titles = list(self.show_data.keys())
        
        # 1. Prepare Data for Indexing
        self.id_to_title = {i: title for i, title in enumerate(self.all_titles)}
        
        sample_vector = next(iter(self.show_data.values()))
        ndim = len(sample_vector)
        
        # 2. Build the Index
        print("Building high-performance vector index (usearch)...")
        self.index = Index(ndim=ndim, metric="cosine")
        
        # Prepare arrays for bulk addition
        keys = np.array(list(self.id_to_title.keys()), dtype=np.longlong)
        vectors = np.array(list(self.show_data.values()), dtype=np.float32)
        
        self.index.add(keys, vectors)
        print(f"Index built with {len(keys)} items.")

    def find_best_matches(self, user_inputs):
        """
        Fuzzy matching for verification (remains unchanged).
        """
        matched_titles = []
        for raw in user_inputs:
            best_match = process.extractOne(raw, self.all_titles)[0]
            matched_titles.append(best_match)
        return matched_titles

    def recommend_shows(self, user_titles, count=5):
        """
        Logic:
        1. Average user vectors.
        2. usearch query (O(1)).
        3. Filter existing shows and format.
        """
        # 1. Average User Vectors
        user_vectors = []
        for t in user_titles:
            if t in self.show_data:
                user_vectors.append(self.show_data[t])
        
        if not user_vectors:
            return []

        avg_vector = np.mean(user_vectors, axis=0).astype(np.float32)
        
        # 2. Search Index
        # We request more results than needed because we might filter out the input shows
        search_limit = count + len(user_titles) + 5
        matches = self.index.search(avg_vector, search_limit)
        
        recommendations = []
        
        # matches.keys = IDs, matches.distances = Cosine Distance
        for doc_id, distance in zip(matches.keys, matches.distances):
            title = self.id_to_title[doc_id]
            
            if title in user_titles:
                continue
            
            # Distance 0.0 = 100% Match
            # Distance 1.0 = 0% Match (Orthogonal)
            score = max(0, (1 - distance) * 100)
            
            recommendations.append((title, int(score)))
            
            if len(recommendations) >= count:
                break
                
        return recommendations