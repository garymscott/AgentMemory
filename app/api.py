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

@app.post("/memories/", response_model=str)
async def create_memory(memory: MemoryCreate, db: Session = Depends(get_db)):
    try:
        # Generate embedding
        embedding = await store.create_embedding(memory.text)
        memory_id = str(uuid.uuid4())

        if memory.session_id:
            # First verify session exists
            session = db.query(DbSession).filter(DbSession.id == memory.session_id).first()
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")

            # Store in Redis for active session
            redis_key = f"session:{memory.session_id}:memory:{memory_id}"
            redis_client.hset(redis_key, mapping={
                "text": memory.text,
                "memory_metadata": str(memory.memory_metadata or {}),
                "embedding": embedding.tobytes()
            })
            redis_client.expire(redis_key, 86400)  # 24 hour TTL
        else:
            # Store directly in PostgreSQL for persistent storage
            db_memory = Memory(
                id=memory_id,
                text=memory.text,
                memory_metadata=memory.memory_metadata or {},
                embedding=embedding,
                session_id=memory.session_id
            )
            db.add(db_memory)
            db.commit()

        return memory_id
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create memory: {str(e)}"
        )

@app.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: str, db: Session = Depends(get_db)):
    # First check Redis for active session memories
    redis_keys = redis_client.keys(f"session:*:memory:{memory_id}")
    if redis_keys:
        memory_data = redis_client.hgetall(redis_keys[0])
        if memory_data:
            session_id = redis_keys[0].split(":")[1]
            return MemoryResponse(
                id=memory_id,
                text=memory_data["text"],
                memory_metadata=eval(memory_data["memory_metadata"]),
                session_id=session_id
            )

    # If not in Redis, check PostgreSQL
    memory = db.query(Memory).filter(Memory.id == memory_id).first()
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    return MemoryResponse(
        id=memory.id,
        text=memory.text,
        memory_metadata=memory.memory_metadata,
        session_id=memory.session_id
    )