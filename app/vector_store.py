import numpy as np
import faiss
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from openai import AsyncOpenAI
import os
from sqlalchemy.orm import Session
from redis import Redis

@dataclass
class Memory:
    id: str
    text: str
    embedding: np.ndarray
    metadata: Dict
    session_id: Optional[str] = None

class VectorStore:
    def __init__(self, dimension: int = 1536, use_faiss: bool = False):
        self.dimension = dimension
        self.use_faiss = use_faiss
        
        # Only initialize OpenAI client if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = AsyncOpenAI()
            self.test_mode = False
        else:
            self.client = None
            self.test_mode = True

        if use_faiss:
            self.index = faiss.IndexFlatIP(self.dimension)

    async def create_embedding(self, text: str) -> np.ndarray:
        """Create embedding using OpenAI API or test embedding"""
        if self.test_mode:
            # Create deterministic test embedding
            seed = sum(ord(c) for c in text)
            np.random.seed(seed)
            return np.random.rand(self.dimension).astype(np.float32)
        else:
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return np.array(response.data[0].embedding, dtype=np.float32)

    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate similarity score between two vectors"""
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)

        # Calculate cosine similarity
        sim = np.dot(vec1_norm, vec2_norm)

        # Apply similarity threshold and scaling
        if sim < 0.75:
            return 0
        else:
            # Scale 0.75-0.85 range to 0-100
            scaled_sim = ((sim - 0.75) / 0.1) * 100
            return min(scaled_sim, 100)  # Cap at 100%