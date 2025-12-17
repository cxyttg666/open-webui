"""add username to auth table

Revision ID: add_username_to_auth
Revises: zzz_make_email_nullable
Create Date: 2025-12-17

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "add_username_to_auth"
down_revision: Union[str, None] = "make_email_nullable_20251215"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Check if column already exists before adding
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [col["name"] for col in inspector.get_columns("auth")]

    if "username" not in columns:
        op.add_column("auth", sa.Column("username", sa.String(), nullable=True))
        # Create unique index for username
        op.create_index("auth_username", "auth", ["username"], unique=True)


def downgrade() -> None:
    op.drop_index("auth_username", table_name="auth")
    op.drop_column("auth", "username")
