# Agent Memory

A semantic memory system for AI agents with working and long-term memory capabilities. Built with FastAPI, Redis, PostgreSQL with pgvector, and FAISS.

## Features

- Dual-layer memory architecture:
  - Working memory (Redis) for active sessions
  - Long-term memory (PostgreSQL + pgvector) for persistent storage
- Session-based memory management
- Semantic search using OpenAI embeddings
- FAISS vector storage for efficient similarity search
- Intuitive similarity scoring (0-100%)
- FastAPI backend with comprehensive API
- Automatic memory persistence and summarization

## Prerequisites

1. Install Miniconda for managing Python environments:
   - Download from [Miniconda website](https://docs.conda.io/en/latest/miniconda.html)
   - Follow installation instructions for your OS

2. Install and start Redis server:
```bash
sudo apt-get update
sudo apt-get install redis
sudo systemctl start redis
```

3. Install PostgreSQL and pgvector:
```bash
sudo apt-get install postgresql
sudo -u postgres psql -c 'CREATE EXTENSION vector;'
```

4. OpenAI API key for generating embeddings

## Setup

1. Clone the repository:
```bash
git clone https://github.com/garymscott/agent-memory.git
cd agent-memory
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yml
conda activate agent-memory
```

3. Set up environment variables:
Create a `.env` file in the root directory:
```
POSTGRES_URL=postgresql://user:password@localhost/agent_memory
REDIS_URL=redis://localhost
OPENAI_API_KEY=your_api_key_here
```

4. Initialize the database:
```bash
createdb agent_memory
psql -d agent_memory -c 'CREATE EXTENSION vector;'
alembic upgrade head
```

## Running the Application

1. Ensure Redis and PostgreSQL are running:
```bash
sudo systemctl status redis
sudo systemctl status postgresql
```

2. Start the backend server:
```bash
# From the root directory
conda activate agent-memory
uvicorn app.api:app --reload
```

## Project Structure

```
agent-memory/
├── alembic/                # Database migrations
├── app/
│   ├── api.py             # FastAPI endpoints
│   ├── database.py        # Database connections
│   ├── models.py          # SQLAlchemy models
│   └── vector_store.py    # Vector operations
├── tests/                 # Test suite
│   ├── conftest.py       # Test configuration
│   ├── test_api.py       # API tests
│   └── test_vector_store.py  # Vector store tests
└── ui/                    # Frontend (coming soon)
```

## API Endpoints

- `POST /sessions/`: Create a new memory session
- `POST /sessions/{session_id}/end`: End a session and persist memories
- `POST /memories/`: Create a new memory
- `GET /memories/`: List all memories
- `GET /memories/{memory_id}`: Get a specific memory
- `POST /memories/search/`: Search for similar memories
- `PUT /memories/{memory_id}`: Update a memory
- `DELETE /memories/{memory_id}`: Delete a memory

## Technical Details

- Uses OpenAI's `text-embedding-ada-002` model for generating embeddings
- Redis for active session storage with TTL
- PostgreSQL with pgvector for persistent storage and similarity search
- Custom similarity scaling:
  - Filters out matches below 0.75 similarity
  - Scales 0.75-0.85 range to 0-100%
  - Provides meaningful differentiation between related and unrelated content

## Development

Run tests:
```bash
# Create test database
createdb agent_memory_test
psql -d agent_memory_test -c 'CREATE EXTENSION vector;'

# Run tests
pytest tests/ -v --asyncio-mode=strict
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT