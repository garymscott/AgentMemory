from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
from vector_store import VectorStore
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
store = VectorStore()

class MemoryCreate(BaseModel):
    text: str
    metadata: Optional[Dict] = None

class MemoryUpdate(BaseModel):
    text: Optional[str] = None
    metadata: Optional[Dict] = None

class MemorySearch(BaseModel):
    query: str
    k: Optional[int] = 5

class MemoryResponse(BaseModel):
    id: str
    text: str
    metadata: Dict
    similarity: Optional[float] = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/memories/", response_model=str)
async def create_memory(memory: MemoryCreate):
    """Create a new memory"""
    try:
        # Add logging to track the flow
        print(f"Received memory: {memory}")  # Debug log

        # Generate embedding
        try:
            embedding = await store.create_embedding(memory.text)
            print(f"Generated embedding of length: {len(embedding)}")  # Debug log
        except Exception as e:
            print(f"Embedding generation failed: {str(e)}")  # Debug log
            raise

        # Add to store
        try:
            memory_id = store.add_memory(
                text=memory.text,
                embedding=embedding,
                metadata=memory.metadata
            )
            print(f"Added memory with ID: {memory_id}")  # Debug log
            return memory_id
        except Exception as e:
            print(f"Store addition failed: {str(e)}")  # Debug log
            raise

    except Exception as e:
        print(f"Top level error: {str(e)}")  # Debug log
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create memory: {str(e)}"
        )

@app.get("/memories/", response_model=List[MemoryResponse])
async def list_memories():
    """List all memories"""
    return [
        MemoryResponse(
            id=memory_id,
            text=memory_data["text"],  # Access dict values with ["key"]
            metadata=memory_data["metadata"]
        )
        for memory_id, memory_data in store.memories.items()
    ]

@app.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: str):
    """Get a specific memory"""
    if memory_id not in store.memories:
        raise HTTPException(status_code=404, detail="Memory not found")

    memory = store.memories[memory_id]
    return MemoryResponse(
        id=memory.id,
        text=memory.text,
        metadata=memory.metadata
    )

@app.post("/memories/search/", response_model=List[MemoryResponse])
async def search_memories(search: MemorySearch):
    """Search for similar memories"""
    try:
        # Log the incoming query
        print(f"Received search query: {search.query}")

        # Generate embedding for query
        query_embedding = await store.create_embedding(search.query)
        print(f"Generated query embedding of length: {len(query_embedding)}")

        # Search store
        results = store.search(query_embedding, k=search.k)
        print(f"Search returned {len(results)} results")

        # Convert to response format
        responses = [
            MemoryResponse(
                id=memory.id,
                text=memory.text,
                metadata=memory.metadata,
                similarity=similarity
            )
            for memory, similarity in results
        ]

        return responses
    except Exception as e:
        print(f"Search error: {str(e)}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/memories/{memory_id}", response_model=bool)
async def update_memory(memory_id: str, update: MemoryUpdate):
    """Update a memory"""
    try:
        # Generate new embedding if text is updated
        embedding = None
        if update.text is not None:
            embedding = await store.create_embedding(update.text)

        success = store.update_memory(
            memory_id=memory_id,
            text=update.text,
            embedding=embedding,
            metadata=update.metadata
        )

        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")

        return success
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memories/{memory_id}", response_model=bool)
async def delete_memory(memory_id: str):
    """Delete a memory"""
    success = store.delete_memory(memory_id)

    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")

    return success
