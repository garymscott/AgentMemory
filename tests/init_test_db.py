import os
from alembic import command
from alembic.config import Config
import logging

logger = logging.getLogger(__name__)

def init_test_database(test_db_url: str):
    """Initialize test database with proper schema"""
    try:
        # Create Alembic configuration
        alembic_cfg = Config("alembic.ini")
        
        # Override the SQLAlchemy URL
        alembic_cfg.set_main_option("sqlalchemy.url", test_db_url)
        
        # Stamp the database with the latest revision
        command.stamp(alembic_cfg, "head")
        
        # Run migrations
        command.upgrade(alembic_cfg, "head")
        
        logger.info("Successfully initialized test database")
        
    except Exception as e:
        logger.error(f"Failed to initialize test database: {str(e)}")
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Get test database URL from environment
    test_db_url = os.getenv(
        "TEST_POSTGRES_URL",
        "postgresql://gary@localhost/agent_memory_test"
    )
    
    # Initialize database
    init_test_database(test_db_url)