"""Fix auth table - remove email column and make username required

Revision ID: fix_auth_table_20251217
Revises: add_username_to_auth
Create Date: 2025-12-17

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "fix_auth_table_20251217"
down_revision: Union[str, None] = "add_username_to_auth"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Remove email column from auth table and ensure username is properly set up"""
    conn = op.get_bind()
    dialect = conn.dialect.name

    if dialect == "sqlite":
        # SQLite requires table rebuild to drop column
        # First, copy data to new table structure
        op.execute("""
            CREATE TABLE auth_new (
                id VARCHAR(255) NOT NULL PRIMARY KEY,
                username VARCHAR(255),
                password TEXT,
                active INTEGER
            )
        """)

        # Copy existing data - use email as username if username is null
        op.execute("""
            INSERT INTO auth_new (id, username, password, active)
            SELECT id, COALESCE(username, email), password, active FROM auth
        """)

        # Drop old table and rename new one
        op.execute("DROP TABLE auth")
        op.execute("ALTER TABLE auth_new RENAME TO auth")

        # Recreate indexes
        op.execute("CREATE UNIQUE INDEX auth_id ON auth (id)")
        op.execute("CREATE UNIQUE INDEX auth_username ON auth (username)")
    else:
        # For PostgreSQL and others
        inspector = sa.inspect(conn)
        columns = [col["name"] for col in inspector.get_columns("auth")]

        if "email" in columns:
            # First migrate email to username if username is null
            op.execute("UPDATE auth SET username = email WHERE username IS NULL")
            op.drop_column("auth", "email")


def downgrade() -> None:
    """Add email column back"""
    conn = op.get_bind()
    dialect = conn.dialect.name

    if dialect == "sqlite":
        op.execute("""
            CREATE TABLE auth_new (
                id VARCHAR(255) NOT NULL PRIMARY KEY,
                email VARCHAR(255),
                username VARCHAR(255),
                password TEXT,
                active INTEGER
            )
        """)

        op.execute("""
            INSERT INTO auth_new (id, email, username, password, active)
            SELECT id, username, username, password, active FROM auth
        """)

        op.execute("DROP TABLE auth")
        op.execute("ALTER TABLE auth_new RENAME TO auth")
        op.execute("CREATE UNIQUE INDEX auth_id ON auth (id)")
        op.execute("CREATE UNIQUE INDEX auth_username ON auth (username)")
    else:
        op.add_column("auth", sa.Column("email", sa.String(), nullable=True))
        op.execute("UPDATE auth SET email = username")
