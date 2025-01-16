from sqlalchemy import Column, String, JSON, DateTime
from pgvector.sqlalchemy import Vector
from datetime import datetime
from .database import Base

class Memory(Base):
    __tablename__ = "memories"

    id = Column(String, primary_key=True)
    text = Column(String, nullable=False)
    memory_metadata = Column(JSON, nullable=False, default={})
    embedding = Column(Vector(1536))
    session_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    status = Column(String, nullable=False)
    session_metadata = Column(JSON, nullable=False, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    summary = Column(String, nullable=True)