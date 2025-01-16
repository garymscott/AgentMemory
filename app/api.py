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

@app.post("/sessions/", response_model=SessionResponse)
async def create_session(session: SessionCreate, db: Session = Depends(get_db)):
    session_id = str(uuid.uuid4())
    db_session = DbSession(
        id=session_id,
        status="active",
        session_metadata=session.session_metadata or {}
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

@app.post("/memories/", response_model=str)
async def create_memory(memory: MemoryCreate, db: Session = Depends(get_db)):
    try:
        if memory.session_id:
            # First verify session exists and is active
            session = db.query(DbSession).filter(
                DbSession.id == memory.session_id,
                DbSession.status == "active"
            ).first()
            if not session:
                raise HTTPException(
                    status_code=404,
                    detail="Active session not found"
                )

        # Generate embedding
        try:
            embedding = await store.create_embedding(memory.text)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embedding: {str(e)}"
            )

        memory_id = str(uuid.uuid4())

        if memory.session_id:
            # Store in Redis for active session
            redis_key = f"session:{memory.session_id}:memory:{memory_id}"
            try:
                redis_client.hset(redis_key, mapping={
                    "text": memory.text,
                    "memory_metadata": str(memory.memory_metadata or {}),
                    "embedding": embedding.tobytes()
                })
                redis_client.expire(redis_key, 86400)  # 24 hour TTL
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to store in Redis: {str(e)}"
                )
        else:
            # Store directly in PostgreSQL
            try:
                db_memory = Memory(
                    id=memory_id,
                    text=memory.text,
                    memory_metadata=memory.memory_metadata or {},
                    embedding=embedding,
                    session_id=memory.session_id
                )
                db.add(db_memory)
                db.commit()
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to store in database: {str(e)}"
                )

        return memory_id

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/memories/", response_model=List[MemoryResponse])
async def list_memories(
    session_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    try:
        if session_id:
            # Verify session exists
            session = db.query(DbSession).filter(DbSession.id == session_id).first()
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")

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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list memories: {str(e)}"
        )

@app.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: str, db: Session = Depends(get_db)):
    try:
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve memory: {str(e)}"
        )