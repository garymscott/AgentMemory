from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
from vector_store import VectorStore

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

@app.post("/memories/", response_model=str)
async def create_memory(memory: MemoryCreate):
    """Create a new memory"""
    try:
        # Generate embedding
        embedding = await store.create_embedding(memory.text)
        
        # Add to store
        memory_id = store.add_memory(
            text=memory.text,
            embedding=embedding,
            metadata=memory.metadata
        )
        
        return memory_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        # Generate query embedding
        query_embedding = await store.create_embedding(search.query)
        
        # Search store
        results = store.search(query_embedding, k=search.k)
        
        return [
            MemoryResponse(
                id=memory.id,
                text=memory.text,
                metadata=memory.metadata,
                similarity=1.0 - dist  # Convert distance to similarity
            )
            for memory, dist in results
        ]
    except Exception as e:
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