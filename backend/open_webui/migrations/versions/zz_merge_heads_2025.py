"""Merge multiple heads

Revision ID: merge_heads_20251215
Revises: 3e0e00844bb0, 6a39f3d8e55c
Create Date: 2025-12-15 22:40:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "merge_heads_20251215"
down_revision: Union[str, Sequence[str], None] = ("3e0e00844bb0", "6a39f3d8e55c")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # This is a merge migration, no changes needed
    pass


def downgrade() -> None:
    # This is a merge migration, no changes needed
    pass
