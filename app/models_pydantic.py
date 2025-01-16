from pydantic import BaseModel
from typing import Dict, Optional
from datetime import datetime

class MemoryCreate(BaseModel):
    text: str
    memory_metadata: Optional[Dict] = {}
    session_id: Optional[str] = None

class MemoryUpdate(BaseModel):
    text: Optional[str] = None
    memory_metadata: Optional[Dict] = {}

class MemorySearch(BaseModel):
    query: str
    k: Optional[int] = 5
    session_id: Optional[str] = None

class MemoryResponse(BaseModel):
    id: str
    text: str
    memory_metadata: Dict = {}
    similarity: Optional[float] = None
    session_id: Optional[str] = None

class SessionCreate(BaseModel):
    session_metadata: Optional[Dict] = {}

class SessionResponse(BaseModel):
    id: str
    status: str
    session_metadata: Dict = {}
    created_at: datetime
    ended_at: Optional[datetime] = None
    summary: Optional[str] = None

    class Config:
        from_attributes = True  # For SQLAlchemy model compatibility