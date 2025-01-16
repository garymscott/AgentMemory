from setuptools import setup, find_packages

setup(
    name="agent-memory",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "redis",
        "psycopg2-binary",
        "sqlalchemy",
        "python-dotenv",
        "openai",
        "numpy",
        "faiss-cpu",
        "pgvector",
        "alembic",
    ],
)