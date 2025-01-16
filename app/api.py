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
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    try:
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
    except Exception as e:
        logger.error(f"Failed to create session: {str(e)}")
        raise

@app.post("/memories/", response_model=str)
async def create_memory(memory: MemoryCreate, db: Session = Depends(get_db)):
    try:
        logger.debug(f"Received create memory request: {memory}")

        if memory.session_id:
            session = db.query(DbSession).filter(
                DbSession.id == memory.session_id,
                DbSession.status == "active"
            ).first()
            if not session:
                logger.error(f"Session not found or not active: {memory.session_id}")
                raise HTTPException(status_code=404, detail="Active session not found")

        logger.debug("Generating embedding...")
        embedding = await store.create_embedding(memory.text)
        memory_id = str(uuid.uuid4())

        if memory.session_id:
            logger.debug(f"Storing memory in Redis for session {memory.session_id}")
            redis_key = f"session:{memory.session_id}:memory:{memory_id}"
            try:
                redis_client.hset(redis_key, mapping={
                    "text": memory.text,
                    "memory_metadata": str(memory.memory_metadata or {}),
                    "embedding": embedding.tobytes()
                })
                redis_client.expire(redis_key, 86400)
            except Exception as e:
                logger.error(f"Redis storage failed: {str(e)}")
                raise
        else:
            logger.debug("Storing memory in PostgreSQL")
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
                logger.error(f"Database storage failed: {str(e)}")
                raise

        return memory_id

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in create_memory: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create memory: {str(e)}"
        )

@app.get("/memories/", response_model=List[MemoryResponse])
async def list_memories(session_id: Optional[str] = None, db: Session = Depends(get_db)):
    try:
        logger.debug(f"Listing memories for session: {session_id}")
        if session_id:
            session = db.query(DbSession).filter(DbSession.id == session_id).first()
            if not session:
                logger.error(f"Session not found: {session_id}")
                raise HTTPException(status_code=404, detail="Session not found")

            redis_keys = redis_client.keys(f"session:{session_id}:memory:*")
            logger.debug(f"Found {len(redis_keys)} Redis keys")
            redis_memories = []
            for key in redis_keys:
                memory_data = redis_client.hgetall(key)
                if memory_data:
                    logger.debug(f"Redis memory data: {memory_data}")
                    memory_id = key.split(":")[-1]
                    redis_memories.append(
                        MemoryResponse(
                            id=memory_id,
                            text=memory_data["text"],
                            memory_metadata=eval(memory_data["memory_metadata"]),
                            session_id=session_id
                        )
                    )

            db_memories = db.query(Memory).filter(Memory.session_id == session_id).all()
            logger.debug(f"Found {len(db_memories)} database memories")
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

    except Exception as e:
        logger.error(f"Error in list_memories: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list memories: {str(e)}"
        )