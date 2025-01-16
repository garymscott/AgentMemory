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

@app.post("/sessions/{session_id}/end", response_model=SessionResponse)
async def end_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(DbSession).filter(DbSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Move session data from Redis to PostgreSQL
    session_keys = redis_client.keys(f"session:{session_id}:memory:*")
    for key in session_keys:
        memory_data = redis_client.hgetall(key)
        if memory_data:
            db_memory = Memory(
                id=str(uuid.uuid4()),
                text=memory_data["text"],
                memory_metadata=eval(memory_data.get("memory_metadata", "{}")),
                embedding=np.frombuffer(memory_data["embedding"]),
                session_id=session_id
            )
            db.add(db_memory)
            redis_client.delete(key)

    session.status = "completed"
    session.ended_at = datetime.utcnow()
    db.commit()
    db.refresh(session)
    return session