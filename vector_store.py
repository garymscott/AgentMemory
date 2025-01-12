import numpy as np
import faiss
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import openai
from pathlib import Path
import pickle

@dataclass
class Memory:
    id: str
    text: str
    embedding: np.ndarray
    metadata: Dict

class VectorStore:
    def __init__(
        self,
        dimension: int = 1536,  # Default for text-embedding-ada-002
        index_key: str = "IVF100,Flat",
        nprobe: int = 10,
        store_path: str = "vector_store"
    ):
        self.dimension = dimension
        self.index_key = index_key
        self.nprobe = nprobe
        self.store_path = Path(store_path)
        self.store_path.mkdir(exist_ok=True)
        
        # Initialize FAISS index
        self.quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFFlat(
            self.quantizer, self.dimension, 100, faiss.METRIC_L2
        )
        
        # Memory management
        self.memories: Dict[str, Memory] = {}
        self.next_id = 0
        
        # Load existing data if available
        self._load_store()

    async def create_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI's API"""
        response = await openai.Embedding.acreate(
            input=text,
            model="text-embedding-ada-002"
        )
        return np.array(response["data"][0]["embedding"], dtype=np.float32)

    def add_memory(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> str:
        """Add a new memory to the store"""
        if not self.index.is_trained:
            # Train with the first vector if index isn't trained
            self.index.train(embedding.reshape(1, -1))
        
        memory_id = str(self.next_id)
        self.next_id += 1
        
        # Store memory
        self.memories[memory_id] = Memory(
            id=memory_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {}
        )
        
        # Add to FAISS index
        self.index.add(embedding.reshape(1, -1))
        
        # Save updates
        self._save_store()
        
        return memory_id

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[Memory, float]]:
        """Search for similar memories"""
        if not self.memories:
            return []
            
        # Set number of clusters to probe
        self.index.nprobe = self.nprobe
        
        # Search index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), k
        )
        
        # Get memories and their distances
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # FAISS returns -1 for empty slots
                memory_id = str(idx)
                if memory_id in self.memories:
                    results.append((self.memories[memory_id], float(dist)))
        
        return results

    def update_memory(
        self,
        memory_id: str,
        text: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Update an existing memory"""
        if memory_id not in self.memories:
            return False
            
        memory = self.memories[memory_id]
        
        # Update fields if provided
        if text is not None:
            memory.text = text
        if embedding is not None:
            memory.embedding = embedding
        if metadata is not None:
            memory.metadata.update(metadata)
            
        # Since FAISS doesn't support updates, we need to rebuild the index
        self._rebuild_index()
        
        # Save updates
        self._save_store()
        
        return True

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory"""
        if memory_id not in self.memories:
            return False
            
        del self.memories[memory_id]
        
        # Rebuild index without deleted memory
        self._rebuild_index()
        
        # Save updates
        self._save_store()
        
        return True

    def _rebuild_index(self):
        """Rebuild FAISS index from scratch"""
        if not self.memories:
            return
            
        # Create new index
        self.quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFFlat(
            self.quantizer, self.dimension, 100, faiss.METRIC_L2
        )
        
        # Get all embeddings
        embeddings = np.vstack([
            memory.embedding for memory in self.memories.values()
        ])
        
        # Train and add vectors
        self.index.train(embeddings)
        self.index.add(embeddings)

    def _save_store(self):
        """Save the vector store to disk"""
        # Save FAISS index
        faiss.write_index(
            self.index,
            str(self.store_path / "index.faiss")
        )
        
        # Save memories and metadata
        memories_data = {
            memory_id: {
                "text": memory.text,
                "metadata": memory.metadata
            }
            for memory_id, memory in self.memories.items()
        }
        
        with open(self.store_path / "memories.pkl", "wb") as f:
            pickle.dump({
                "memories": memories_data,
                "next_id": self.next_id
            }, f)

    def _load_store(self):
        """Load the vector store from disk"""
        index_path = self.store_path / "index.faiss"
        memories_path = self.store_path / "memories.pkl"
        
        if index_path.exists() and memories_path.exists():
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load memories and metadata
            with open(memories_path, "rb") as f:
                data = pickle.load(f)
                
            memories_data = data["memories"]
            self.next_id = data["next_id"]
            
            # Reconstruct memories
            for memory_id, memory_data in memories_data.items():
                # Get embedding from FAISS index
                idx = int(memory_id)
                embedding = self._get_embedding_from_index(idx)
                
                self.memories[memory_id] = Memory(
                    id=memory_id,
                    text=memory_data["text"],
                    embedding=embedding,
                    metadata=memory_data["metadata"]
                )

    def _get_embedding_from_index(self, idx: int) -> np.ndarray:
        """Extract embedding from FAISS index by ID"""
        return self.index.reconstruct(idx)