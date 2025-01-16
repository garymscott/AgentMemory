import pytest
from fastapi.testclient import TestClient

def test_create_session(client):
    # Create test data
    test_data = {
        "session_metadata": {"purpose": "test session"}
    }

    # Send request
    response = client.post("/sessions/", json=test_data)

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert data["session_metadata"].get("purpose") == "test session"
    return data["id"]

def test_create_memory(client):
    # First create a session
    session_id = test_create_session(client)

    # Create a memory within the session
    memory_data = {
        "text": "Test memory",
        "memory_metadata": {"tag": "test"},
        "session_id": session_id
    }
    response = client.post("/memories/", json=memory_data)
    assert response.status_code == 200
    memory_id = response.json()
    assert isinstance(memory_id, str)
    return memory_id, session_id

def test_list_memories(client):
    # Create a memory first
    memory_id, session_id = test_create_memory(client)

    # List all memories
    response = client.get("/memories/")
    assert response.status_code == 200
    memories = response.json()
    assert isinstance(memories, list)
    if memories:  # Only check if there are memories
        assert "text" in memories[0]
        assert "memory_metadata" in memories[0]

    # List memories for specific session
    response = client.get(f"/memories/?session_id={session_id}")
    assert response.status_code == 200
    session_memories = response.json()
    assert isinstance(session_memories, list)
    assert len(session_memories) > 0
    assert session_memories[0]["session_id"] == session_id

def test_get_memory(client):
    # Create a memory first
    memory_id, session_id = test_create_memory(client)

    # Get the memory
    response = client.get(f"/memories/{memory_id}")
    assert response.status_code == 200
    memory = response.json()
    assert memory["id"] == memory_id
    assert memory["text"] == "Test memory"
    assert memory["memory_metadata"]["tag"] == "test"
    assert memory["session_id"] == session_id

def test_search_memories(client):
    # Create some test memories
    memory_id, session_id = test_create_memory(client)

    # Search for memories
    search_data = {
        "query": "Test memory",
        "k": 5,
        "session_id": session_id
    }
    response = client.post("/memories/search/", json=search_data)
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    if len(results) > 0:
        assert "similarity" in results[0]
        assert "text" in results[0]
        assert "memory_metadata" in results[0]

def test_update_memory(client):
    # Create a memory first
    memory_id, _ = test_create_memory(client)

    # Update the memory
    update_data = {
        "text": "Updated test memory",
        "memory_metadata": {"tag": "updated"}
    }
    response = client.put(f"/memories/{memory_id}", json=update_data)
    assert response.status_code == 200
    assert response.json() is True

    # Verify the update
    response = client.get(f"/memories/{memory_id}")
    assert response.status_code == 200
    memory = response.json()
    assert memory["text"] == "Updated test memory"
    assert memory["memory_metadata"]["tag"] == "updated"

def test_delete_memory(client):
    # Create a memory first
    memory_id, _ = test_create_memory(client)

    # Delete the memory
    response = client.delete(f"/memories/{memory_id}")
    assert response.status_code == 200
    assert response.json() is True

    # Verify the deletion
    response = client.get(f"/memories/{memory_id}")
    assert response.status_code == 404

def test_end_session(client):
    # Create a session with some memories
    memory_id, session_id = test_create_memory(client)

    # End the session
    response = client.post(f"/sessions/{session_id}/end")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["ended_at"] is not None

def test_error_handling(client):
    # Test non-existent memory
    response = client.get("/memories/nonexistent-id")
    assert response.status_code == 404

    # Test invalid session
    response = client.post("/sessions/nonexistent-id/end")
    assert response.status_code == 404

    # Test invalid memory update
    response = client.put("/memories/nonexistent-id", json={"text": "test"})
    assert response.status_code == 404

    # Test invalid memory deletion
    response = client.delete("/memories/nonexistent-id")
    assert response.status_code == 404