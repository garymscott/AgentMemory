"""create initial tables

Revision ID: 20240116_initial
Revises: 
Create Date: 2024-01-16
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic
revision: str = '20240116_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Create sessions table
    op.create_table(
        'sessions',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('session_metadata', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('ended_at', sa.DateTime(), nullable=True),
        sa.Column('summary', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create memories table
    op.create_table(
        'memories',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('text', sa.String(), nullable=False),
        sa.Column('memory_metadata', sa.JSON(), nullable=False),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.Column('session_id', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade() -> None:
    op.drop_table('memories')
    op.drop_table('sessions')