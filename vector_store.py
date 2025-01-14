import numpy as np
import faiss
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from openai import AsyncOpenAI
from pathlib import Path
import pickle
from dotenv import load_dotenv
import uuid

load_dotenv()

@dataclass
class Memory:
    id: str
    text: str
    embedding: np.ndarray
    metadata: Dict

class VectorStore:
    def __init__(self, dimension: int = 1536, store_path: str = "vector_store"):
        self.client = AsyncOpenAI()
        self.dimension = dimension
        self.store_path = Path(store_path)
        self.store_path.mkdir(exist_ok=True)

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)

        # Memory management
        self.memories = {}
        self.id_to_index = {}  # UUID to FAISS index mapping
        self.index_to_id = {}  # FAISS index to UUID mapping
        self.next_index = 0    # Track next available FAISS index

        # Load existing data
        self._load_store()

    async def create_embedding(self, text: str) -> list[float]:
        """Create embedding using OpenAI API"""
        response = await self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

    def add_memory(self, text: str, embedding: list[float], metadata: dict = None) -> str:
        """Add a memory to the store"""
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding_array) # In-place normalization
        memory_id = str(uuid.uuid4())

        # Add to FAISS index
        self.index.add(embedding_array)

        # Update mappings
        self.id_to_index[memory_id] = self.next_index
        self.index_to_id[self.next_index] = memory_id
        self.next_index += 1

        # Store memory
        self.memories[memory_id] = {
            "text": text,
            "metadata": metadata
        }

        self._save_store()
        return memory_id

    def search(self, query_embedding: list[float], k: int = 5) -> List[Tuple[Memory, float]]:
        try:
            query_array = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query_array)

            similarities, indices = self.index.search(query_array, k)

            results = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx == -1:
                    continue

                memory_id = self.index_to_id.get(int(idx))
                if memory_id and memory_id in self.memories:
                    memory_data = self.memories[memory_id]
                    embedding = self.index.reconstruct(int(idx)).tolist()

                    # New scaling approach:
                    # - Below 0.75: 0%
                    # - 0.75 to 0.85: Linear scaling from 0-100%
                    if sim < 0.75:
                        scaled_sim = 0
                    else:
                        # Scale 0.75-0.85 range to 0-100
                        scaled_sim = ((sim - 0.75) / 0.1) * 100
                        scaled_sim = min(scaled_sim, 100)  # Cap at 100%

                    if scaled_sim > 0:
                        print(f"Memory: {memory_data['text']}")
                        print(f"Raw similarity: {sim:.4f}, Scaled: {scaled_sim:.1f}%")

                        memory = Memory(
                            id=memory_id,
                            text=memory_data["text"],
                            metadata=memory_data["metadata"],
                            embedding=embedding
                        )
                        results.append((memory, float(scaled_sim)))

            return results

        except Exception as e:
            print(f"Search error: {str(e)}")
            raise

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
        self.index = faiss.IndexFlatL2(self.dimension)

        # Get all embeddings
        embeddings = np.vstack([
            memory.embedding for memory in self.memories.values()
        ])

        # Train and add vectors
        self.index.add(embeddings)


    def _save_store(self):
        """Save the vector store to disk"""
        # Save FAISS index
        faiss.write_index(self.index, str(self.store_path / "index.faiss"))

        # Save memories and mappings
        save_data = {
            "memories": {
                memory_id: {
                    "text": memory["text"],
                    "metadata": memory["metadata"]
                }
                for memory_id, memory in self.memories.items()
            },
            "id_to_index": self.id_to_index,
            "next_index": self.next_index
        }

        with open(self.store_path / "memories.pkl", "wb") as f:
            pickle.dump(save_data, f)

    def _load_store(self):
        """Load the vector store from disk"""
        index_path = self.store_path / "index.faiss"
        memories_path = self.store_path / "memories.pkl"

        if not (index_path.exists() and memories_path.exists()):
            return

        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))

            # Load memories and mappings
            with open(memories_path, "rb") as f:
                data = pickle.load(f)

            # Handle old format data
            if isinstance(data, dict) and "memories" in data:
                if "id_to_index" in data:
                    # New format
                    self.memories = data["memories"]
                    self.id_to_index = data["id_to_index"]
                    self.next_index = data.get("next_index", 0)
                else:
                    # Old format - rebuild mappings
                    print("Converting old format data to new format")
                    self.memories = data["memories"]
                    self.id_to_index = {}
                    for idx, memory_id in enumerate(self.memories.keys()):
                        self.id_to_index[memory_id] = idx
                    self.next_index = len(self.memories)

            # Rebuild index_to_id mapping
            self.index_to_id = {idx: id for id, idx in self.id_to_index.items()}

        except Exception as e:
            print(f"Error loading store: {str(e)}")
            # Initialize empty index if load fails
            self.index = faiss.IndexFlatL2(self.dimension)
            self.memories = {}
            self.id_to_index = {}
            self.index_to_id = {}
            self.next_index = 0

    def _get_embedding_from_index(self, idx: int) -> np.ndarray:
        """Extract embedding from FAISS index by ID"""
        return self.index.reconstruct(idx)
