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
        self.client = AsyncOpenAI()
        self.dimension = dimension
        self.use_faiss = use_faiss

        if use_faiss:
            # Initialize FAISS index for optional in-memory search
            self.index = faiss.IndexFlatIP(self.dimension)

    async def create_embedding(self, text: str) -> np.ndarray:
        """Create embedding using OpenAI API"""
        response = await self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def add_to_faiss(self, memory_id: str, embedding: np.ndarray):
        """Add embedding to FAISS index if enabled"""
        if self.use_faiss:
            embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(embedding_array)
            self.index.add(embedding_array)
            return self.index.ntotal - 1  # Return the index
        return None

    async def search_similar(
        self,
        query: str,
        db: Session,
        redis_client: Redis,
        session_id: Optional[str] = None,
        k: int = 5
    ) -> List[Tuple[Memory, float]]:
        """Search for similar memories using pgvector and/or Redis"""
        # Generate query embedding
        query_embedding = await self.create_embedding(query)
        
        results = []

        if session_id:
            # Search in Redis for active session
            redis_keys = redis_client.keys(f"session:{session_id}:memory:*")
            for key in redis_keys:
                memory_data = redis_client.hgetall(key)
                if memory_data:
                    memory_embedding = np.frombuffer(
                        memory_data["embedding"],
                        dtype=np.float32
                    )
                    similarity = self._calculate_similarity(
                        query_embedding,
                        memory_embedding
                    )
                    if similarity > 0:
                        memory = Memory(
                            id=key.split(":")[-1],
                            text=memory_data["text"],
                            embedding=memory_embedding,
                            metadata=eval(memory_data["memory_metadata"]),
                            session_id=session_id
                        )
                        results.append((memory, similarity))

        # Search in PostgreSQL with pgvector
        from app.models import Memory as DbMemory
        db_memories = db.query(DbMemory).order_by(
            DbMemory.embedding.cosine_distance(query_embedding)
        ).limit(k).all()

        for db_memory in db_memories:
            similarity = self._calculate_similarity(
                query_embedding,
                np.array(db_memory.embedding)
            )
            if similarity > 0:
                memory = Memory(
                    id=db_memory.id,
                    text=db_memory.text,
                    embedding=db_memory.embedding,
                    metadata=db_memory.memory_metadata,
                    session_id=db_memory.session_id
                )
                results.append((memory, similarity))

        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

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