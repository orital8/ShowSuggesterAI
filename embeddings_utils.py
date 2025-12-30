import numpy as np

def cosine_similarity(a, b):
    """
    Returns the cosine similarity between two vectors.
    1.0 = Identical
    0.0 = Orthogonal
    -1.0 = Opposite
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))