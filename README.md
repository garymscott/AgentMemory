# AgentMemory

A vector storage system for AI agents using FAISS, designed for efficient storage and retrieval of embeddings with a FastAPI interface.

## Features

- FAISS-based vector storage with IVFFlat index
- FastAPI REST interface
- OpenAI embeddings integration
- Docker deployment support
- Prometheus monitoring
- Persistent storage

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/garymscott/AgentMemory.git
cd AgentMemory
```

2. Create a `.env` file:
```bash
OPENAI_API_KEY=your_api_key_here
ENVIRONMENT=development
LOG_LEVEL=debug
```

3. Start with Docker Compose:
```bash
docker-compose up --build
```

4. The API will be available at `http://localhost:8000`

## License

MIT