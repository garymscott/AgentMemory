import numpy as np
from typing import List

class MockVectorStore:
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    async def create_embedding(self, text: str) -> np.ndarray:
        """Create a mock embedding of the specified dimension"""
        # Create a deterministic but unique embedding based on the text
        # This ensures consistent test behavior
        seed = sum(ord(c) for c in text)
        np.random.seed(seed)
        return np.random.rand(self.dimension).astype(np.float32)

    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate mock similarity between two vectors"""
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # Calculate cosine similarity
        sim = np.dot(vec1_norm, vec2_norm)
        
        # Apply similarity threshold and scaling
        if sim < 0.75:
            return 0
        else:
            # Scale 0.75-1.0 range to 0-100
            scaled_sim = ((sim - 0.75) / 0.25) * 100
            return min(scaled_sim, 100)