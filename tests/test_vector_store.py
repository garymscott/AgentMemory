import pytest
import numpy as np
from app.vector_store import VectorStore
from app.models import Memory
from sqlalchemy.orm import Session

@pytest.fixture
def vector_store():
    return VectorStore(dimension=1536)

async def test_create_embedding(vector_store):
    text = "Test memory"
    embedding = await vector_store.create_embedding(text)
    assert isinstance(embedding, list)
    assert len(embedding) == 1536  # OpenAI embedding dimension

async def test_search_similar(vector_store, db_session):
    # Create some test memories
    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "A lazy dog sleeps while the fox runs"
    text3 = "Something completely different about cats"

    # Create embeddings
    embedding1 = await vector_store.create_embedding(text1)
    embedding2 = await vector_store.create_embedding(text2)
    embedding3 = await vector_store.create_embedding(text3)

    # Store in database
    session_id = "test-session"
    memories = [
        Memory(
            id=f"test-{i}",
            text=text,
            embedding=embedding,
            session_id=session_id,
            memory_metadata={}
        )
        for i, (text, embedding) in enumerate([
            (text1, embedding1),
            (text2, embedding2),
            (text3, embedding3)
        ])
    ]

    for memory in memories:
        db_session.add(memory)
    db_session.commit()

    # Search for similar memories
    results = await vector_store.search_similar(
        "fox jumping",
        db_session,
        test_redis_client,
        session_id,
        k=2
    )

    assert len(results) > 0
    # First result should be more similar to the query about fox
    assert "fox" in results[0][0].text.lower()

def test_calculate_similarity(vector_store):
    # Create two similar vectors
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0.9, 0.1, 0])

    similarity = vector_store._calculate_similarity(vec1, vec2)
    assert 0 <= similarity <= 100

    # Test with identical vectors
    similarity = vector_store._calculate_similarity(vec1, vec1)
    assert similarity == 100

    # Test with orthogonal vectors
    vec3 = np.array([0, 1, 0])
    similarity = vector_store._calculate_similarity(vec1, vec3)
    assert similarity == 0