import pickle
import os

PICKLE_PATH = "show_embeddings.pkl"

def verify():
    if not os.path.exists(PICKLE_PATH):
        print(f"FAILED: {PICKLE_PATH} does not exist. Please run preprocess_embeddings.py first.")
        return

    with open(PICKLE_PATH, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Success! Loaded embeddings for {len(data)} shows.")
    
    # Check one entry structure
    first_title = list(data.keys())[0]
    vector = data[first_title]
    print(f"Sample Entry: '{first_title}' has a vector of length {len(vector)}")

if __name__ == "__main__":
    verify()