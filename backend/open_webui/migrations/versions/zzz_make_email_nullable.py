"""Make email field nullable

Revision ID: make_email_nullable_20251215
Revises: merge_heads_20251215
Create Date: 2025-12-15 23:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "make_email_nullable_20251215"
down_revision: Union[str, None] = "merge_heads_20251215"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Make email column nullable in user table"""
    conn = op.get_bind()
    dialect = conn.dialect.name

    if dialect == "sqlite":
        # SQLite requires table rebuild
        with op.batch_alter_table("user", schema=None) as batch_op:
            batch_op.alter_column(
                "email",
                existing_type=sa.String(),
                nullable=True,
            )
    elif dialect == "postgresql":
        # PostgreSQL can alter directly
        op.alter_column("user", "email", existing_type=sa.String(), nullable=True)
    else:
        # For other databases, try generic approach
        op.alter_column("user", "email", existing_type=sa.String(), nullable=True)


def downgrade() -> None:
    """Make email column NOT nullable again"""
    conn = op.get_bind()
    dialect = conn.dialect.name

    if dialect == "sqlite":
        with op.batch_alter_table("user", schema=None) as batch_op:
            batch_op.alter_column(
                "email",
                existing_type=sa.String(),
                nullable=False,
            )
    elif dialect == "postgresql":
        op.alter_column("user", "email", existing_type=sa.String(), nullable=False)
    else:
        op.alter_column("user", "email", existing_type=sa.String(), nullable=False)
