import pickle
import os
import numpy as np
from thefuzz import process
# Import the math function from your simplified utils file
from embeddings_utils import cosine_similarity

class TVShowRecommender:
    def __init__(self, pickle_path="show_embeddings.pkl"):
        """
        Initialize the recommender by loading the cached embeddings.
        """
        if not os.path.exists(pickle_path):
             raise FileNotFoundError(f"Pickle file not found at {pickle_path}. Please run preprocess_embeddings.py first.")
             
        print(f"Loading data from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            self.show_data = pickle.load(f)
            
        self.all_titles = list(self.show_data.keys())

    def find_best_matches(self, user_inputs):
        """
        Takes a list of raw user strings and returns a list of the 
        best-matching official titles from our dataset.
        """
        matched_titles = []
        for raw in user_inputs:
            # process.extractOne returns (BestMatchString, Score)
            best_match = process.extractOne(raw, self.all_titles)[0]
            matched_titles.append(best_match)
        
        return matched_titles

    def recommend_shows(self, user_titles, count=5):
        """
        Logic:
        1. Average the vectors of user_titles.
        2. Loop over all shows to find shortest distance (highest similarity).
        3. Return top 'count' results with match percentage.
        """
        # 1. Get vectors for user shows
        user_vectors = []
        for t in user_titles:
            if t in self.show_data:
                user_vectors.append(self.show_data[t])
        
        if not user_vectors:
            return []

        # Calculate Average Vector
        avg_vector = np.mean(user_vectors, axis=0)
        
        # 2. Loop over all shows and calculate similarity
        scores = []
        for title, vector in self.show_data.items():
            # Skip the shows the user already input
            if title in user_titles:
                continue
            
            # Using the imported math function
            similarity = cosine_similarity(avg_vector, vector)
            scores.append((title, similarity))
            
        # 3. Sort by similarity (Highest first)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N
        top_matches = scores[:count]
        
        # Format the output with percentage
        recommendations = []
        for title, sim_score in top_matches:
            percent = int(sim_score * 100)
            recommendations.append((title, percent))
            
        return recommendations