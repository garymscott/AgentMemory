from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
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
store = VectorStore()  # Keeping the existing vector store for compatibility


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session management endpoints
@app.post("/sessions/", response_model=SessionResponse)
async def create_session(session: SessionCreate, db: Session = Depends(get_db)):
    """Create a new memory session"""
    session_id = str(uuid.uuid4())
    db_session = DbSession(
        id=session_id,
        status="active",
        session_metadata=session.session_metadata or {}  # Ensure we never store None
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

@app.post("/sessions/{session_id}/end", response_model=SessionResponse)
async def end_session(session_id: str, db: Session = Depends(get_db)):
    """End a session and generate summary"""
    session = db.query(DbSession).filter(DbSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Move session data from Redis to PostgreSQL
    session_keys = redis_client.keys(f"session:{session_id}:*")
    for key in session_keys:
        memory_data = redis_client.hgetall(key)
        if memory_data:
            # Create new memory in PostgreSQL
            db_memory = Memory(
                id=str(uuid.uuid4()),
                text=memory_data["text"],
                metadata=memory_data.get("metadata", {}),
                embedding=memory_data.get("embedding"),
                session_id=session_id
            )
            db.add(db_memory)
            # Delete from Redis
            redis_client.delete(key)

    # Update session status
    session.status = "completed"
    session.ended_at = datetime.utcnow()
    # TODO: Add summary generation logic here
    db.commit()
    db.refresh(session)
    return session

@app.post("/memories/", response_model=str)
async def create_memory(memory: MemoryCreate, db: Session = Depends(get_db)):
    """Create a new memory"""
    try:
        # Generate embedding
        embedding = await store.create_embedding(memory.text)
        memory_id = str(uuid.uuid4())

        if memory.session_id:
            # Store in Redis for active session
            redis_key = f"session:{memory.session_id}:memory:{memory_id}"
            redis_client.hset(redis_key, mapping={
                "text": memory.text,
                "metadata": str(memory.session_metadata or {}),
                "embedding": embedding.tobytes()
            })
            # Set TTL for Redis entry (e.g., 24 hours)
            redis_client.expire(redis_key, 86400)
        else:
            # Store directly in PostgreSQL for persistent storage
            db_memory = Memory(
                id=memory_id,
                text=memory.text,
                metadata=memory.session_metadata or {},
                embedding=embedding,
                session_id=memory.session_id
            )
            db.add(db_memory)
            db.commit()

        return memory_id
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create memory: {str(e)}"
        )

@app.get("/memories/", response_model=List[MemoryResponse])
async def list_memories(
    session_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all memories"""
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
                        metadata=eval(memory_data["memory_metadata"]),
                        session_id=session_id
                    )
                )

        # Get memories from PostgreSQL for this session
        db_memories = db.query(Memory).filter(Memory.session_id == session_id).all()
        db_responses = [
            MemoryResponse(
                id=memory.id,
                text=memory.text,
                metadata=memory.session_metadata,
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
                metadata=memory.session_metadata,
                session_id=memory.session_id
            )
            for memory in memories
        ]

@app.post("/memories/search/", response_model=List[MemoryResponse])
async def search_memories(
    search: MemorySearch,
    db: Session = Depends(get_db)
):
    """Search for similar memories"""
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
                            metadata=eval(memory_data["memory_metadata"]),
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
                    metadata=memory.metadata,
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

# Keep existing update and delete endpoints, but update them to work with both storages
@app.put("/memories/{memory_id}", response_model=bool)
async def update_memory(
    memory_id: str,
    update: MemoryUpdate,
    db: Session = Depends(get_db)
):
    """Update a memory"""
    try:
        # Check Redis first
        redis_keys = redis_client.keys(f"session:*:memory:{memory_id}")
        if redis_keys:
            key = redis_keys[0]
            if update.text:
                embedding = await store.create_embedding(update.text)
                redis_client.hset(key, "text", update.text)
                redis_client.hset(key, "embedding", embedding.tobytes())
            if update.memory_metadata:
                redis_client.hset(key, "memory_metadata", str(update.memory_metadata))
            return True

        # If not in Redis, update in PostgreSQL
        memory = db.query(Memory).filter(Memory.id == memory_id).first()
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")

        if update.text:
            memory.text = update.text
            memory.embedding = await store.create_embedding(update.text)
        if update.memory_metadata:
            memory.memory_metadata = update.memory_metadata

        db.commit()
        return True

    except HTTPException:
        raise  # Re-raise HTTP exceptions (like 404)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update memory: {str(e)}"
        )

@app.delete("/memories/{memory_id}", response_model=bool)
async def delete_memory(memory_id: str, db: Session = Depends(get_db)):
    """Delete a memory"""
    # Check Redis first
    redis_keys = redis_client.keys(f"session:*:memory:{memory_id}")
    if redis_keys:
        redis_client.delete(redis_keys[0])
        return True

    # If not in Redis, delete from PostgreSQL
    memory = db.query(Memory).filter(Memory.id == memory_id).first()
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    db.delete(memory)
    db.commit()
    return True

@app.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: str, db: Session = Depends(get_db)):
    """Get a specific memory"""
    # First check Redis for active session memories
    redis_keys = redis_client.keys(f"session:*:memory:{memory_id}")
    if redis_keys:
        memory_data = redis_client.hgetall(redis_keys[0])
        if memory_data:
            session_id = redis_keys[0].split(":")[1]  # Extract session_id from key
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