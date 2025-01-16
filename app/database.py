from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from redis import Redis
import os
from alembic import command
from alembic.config import Config

# Database URLs
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://gary@localhost/agent_memory"
)
REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost/0"
)

# Create engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Redis client
redis_client = Redis.from_url(REDIS_URL, decode_responses=True)

# Create base class for SQLAlchemy models
Base = declarative_base()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to run database migrations
def run_migrations(connection_url: str):
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", connection_url)
    command.upgrade(alembic_cfg, "head")