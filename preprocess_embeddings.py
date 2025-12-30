import os
import pickle
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
CSV_PATH = "tv_shows.csv" 
PICKLE_PATH = "show_embeddings.pkl"
API_KEY = os.getenv("OPENAI_API_KEY")

def generate_embeddings():
    # 1. Validation
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")
    
    # 2. Load Data
    print(f"Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # 3. Setup Client
    client = OpenAI(api_key=API_KEY)
    
    show_embeddings = {}
    print(f"Generating embeddings for {len(df)} shows using text-embedding-3-small...")

    # 4. Generate Embeddings
    for index, row in df.iterrows():
        title = row['Title']
        description = row['Description']
        
        # Skip empty descriptions
        if pd.isna(description) or str(description).strip() == "":
            print(f"Skipping {title}: No description found.")
            continue
            
        try:
            # Call API
            response = client.embeddings.create(
                input=description,
                model="text-embedding-3-small"
            )
            
            vector = response.data[0].embedding
            show_embeddings[title] = vector
            
            # Optional: Print progress every 10 shows to reduce clutter
            if index % 10 == 0:
                print(f"Processed {index}/{len(df)}: {title}")
                
        except Exception as e:
            # Let it crash for major errors, but maybe we want to know which one failed
            print(f"Error processing {title}: {e}")
            raise e

    # 5. Save to Pickle
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(show_embeddings, f)
        
    print(f"Success! Embeddings saved to {PICKLE_PATH}")

if __name__ == "__main__":
    generate_embeddings()