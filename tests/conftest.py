import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base, get_db
from app.api import app
import os
from redis import Redis
from dotenv import load_dotenv
import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load test environment variables
load_dotenv('.env.test')

# Test database URL
TEST_POSTGRES_URL = os.getenv(
    "TEST_POSTGRES_URL",
    "postgresql://gary@localhost/agent_memory_test"
)
TEST_REDIS_URL = os.getenv(
    "TEST_REDIS_URL",
    "redis://localhost/1"
)

# Create test database engine
engine = create_engine(TEST_POSTGRES_URL, echo=True)  # Enable SQL logging
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Test Redis client
test_redis_client = Redis.from_url(TEST_REDIS_URL, decode_responses=True)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database session for each test."""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture
def client(db_session):
    """Create a test client with a fresh database session."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture(autouse=True)
def clean_redis():
    """Clean Redis database after each test."""
    yield
    test_redis_client.flushdb()