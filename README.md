A semantic memory system that allows storage and retrieval of memories using natural language understanding. Built with FastAPI, FAISS vector storage, and Next.js.

## Features

- Semantic search using OpenAI embeddings
- FAISS vector storage for efficient similarity search
- Intuitive similarity scoring (0-100%)
- FastAPI backend
- Next.js frontend with real-time search
- Persistent storage of memories and embeddings

## Prerequisites

1. Install Miniconda for managing Python environments:
   - Download from [Miniconda website](https://docs.conda.io/en/latest/miniconda.html)
   - Follow installation instructions for your OS

2. OpenAI API key for generating embeddings

## Setup

1. Clone the repository:
```bash
git clone https://github.com/garymscott/agent-memory.git
cd agent-memory
```

2. Create and activate conda environment:
```bash
conda create -n agent-memory python=3.10
conda activate agent-memory
```

3. Install FAISS using conda:
```bash
conda install -c conda-forge faiss-cpu
```

4. Install other Python dependencies:
```bash
pip install fastapi uvicorn python-dotenv openai
```

5. Set up environment variables:
   Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_api_key_here
```

6. Install frontend dependencies:
```bash
cd ui
npm install
```

7. Set up frontend environment:
   Create `ui/.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Running the Application

1. Start the backend server:
```bash
# From the root directory
conda activate agent-memory
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

2. Start the frontend development server:
```bash
# From the ui directory
npm run dev
```

3. Access the application at `http://localhost:3000`

## Usage

1. **Adding Memories**:
   - Enter text in the memory input field
   - Add optional metadata like Location, Occasion, etc.
   - Click \"Add Memory\" to store

2. **Searching Memories**:
   - Type in the search box
   - Results appear automatically with similarity scores
   - Matches are based on semantic meaning, not just keywords

## Project Structure

```
agent-memory/
├── api.py              # FastAPI backend
├── vector_store.py     # FAISS vector store implementation
├── ui/                 # Next.js frontend
│   ├── components/     # React components
│   ├── lib/           # Utility functions
│   └── app/           # Next.js pages
└── vector_store/      # Persistent storage directory
    ├── index.faiss    # Vector embeddings
    └── memories.pkl   # Memory texts and metadata
```

## Technical Details

- Uses OpenAI's `text-embedding-ada-002` model for generating embeddings
- FAISS IndexFlatIP for similarity search
- Custom similarity scaling for intuitive results:
  - Filters out matches below 0.75 similarity
  - Scales 0.75-0.85 range to 0-100%
  - Provides meaningful differentiation between related and unrelated content