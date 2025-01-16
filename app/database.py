from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from redis import Redis
import os
from dotenv import load_dotenv

load_dotenv()

# Database URLs
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:password@localhost/agent_memory")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost")

# PostgreSQL connection
engine = create_engine(POSTGRES_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Redis connection
redis_client = Redis.from_url(REDIS_URL, decode_responses=True)

# SQLAlchemy Base
Base = declarative_base()

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()