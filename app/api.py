from fastapi import FastAPI, HTTPException, Depends
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime
import uuid
from sqlalchemy.orm import Session
from app.vector_store import VectorStore
from fastapi.middleware.cors import CORSMiddleware
from app.database import get_db, redis_client
from app.models import Memory, Session as DbSession
from app.models_pydantic import (
    MemoryCreate, MemoryUpdate, MemorySearch, MemoryResponse,
    SessionCreate, SessionResponse
)

app = FastAPI()
store = VectorStore()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/memories/", response_model=List[MemoryResponse])
async def list_memories(
    session_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    if session_id:
        # Check Redis first for active session
        redis_keys = redis_client.keys(f"session:{session_id}:memory:*")
        redis_memories = []
        for key in redis_keys:
            memory_data = redis_client.hgetall(key)
            if memory_data:
                memory_id = key.split(":")[-1]
                redis_memories.append(
                    MemoryResponse(
                        id=memory_id,
                        text=memory_data["text"],
                        memory_metadata=eval(memory_data["memory_metadata"]),
                        session_id=session_id
                    )
                )

        # Get memories from PostgreSQL for this session
        db_memories = db.query(Memory).filter(Memory.session_id == session_id).all()
        db_responses = [
            MemoryResponse(
                id=memory.id,
                text=memory.text,
                memory_metadata=memory.memory_metadata,
                session_id=memory.session_id
            )
            for memory in db_memories
        ]

        return redis_memories + db_responses
    else:
        # Only return persistent memories from PostgreSQL
        memories = db.query(Memory).all()
        return [
            MemoryResponse(
                id=memory.id,
                text=memory.text,
                memory_metadata=memory.memory_metadata,
                session_id=memory.session_id
            )
            for memory in memories
        ]

@app.post("/memories/search/", response_model=List[MemoryResponse])
async def search_memories(
    search: MemorySearch,
    db: Session = Depends(get_db)
):
    try:
        query_embedding = await store.create_embedding(search.query)
        results = []

        if search.session_id:
            # Search in Redis for active session
            redis_keys = redis_client.keys(f"session:{search.session_id}:memory:*")
            for key in redis_keys:
                memory_data = redis_client.hgetall(key)
                if memory_data:
                    memory_embedding = np.frombuffer(memory_data["embedding"])
                    similarity = np.dot(query_embedding, memory_embedding)
                    results.append((
                        MemoryResponse(
                            id=key.split(":")[-1],
                            text=memory_data["text"],
                            memory_metadata=eval(memory_data["memory_metadata"]),
                            similarity=float(similarity),
                            session_id=search.session_id
                        ),
                        similarity
                    ))

        # Search in PostgreSQL using pgvector
        db_results = db.query(Memory).order_by(
            Memory.embedding.cosine_distance(query_embedding)
        ).limit(search.k).all()

        for memory in db_results:
            similarity = 1 - np.dot(query_embedding, memory.embedding)
            results.append((
                MemoryResponse(
                    id=memory.id,
                    text=memory.text,
                    memory_metadata=memory.memory_metadata,
                    similarity=float(similarity),
                    session_id=memory.session_id
                ),
                similarity
            ))

        # Sort combined results by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return [result[0] for result in results[:search.k]]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))