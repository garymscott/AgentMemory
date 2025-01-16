import os
from alembic import command
from alembic.config import Config
from app.database import DATABASE_URL

def init_test_database(test_db_url: str):
    """Initialize test database with proper schema"""
    # Create Alembic configuration
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", test_db_url)
    
    # Run migrations
    command.upgrade(alembic_cfg, "head")