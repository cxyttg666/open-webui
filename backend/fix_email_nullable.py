#!/usr/bin/env python3
"""Fix user table to make email nullable"""
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from open_webui.internal.db import engine
from sqlalchemy import text, inspect

print("Checking database...")
inspector = inspect(engine)
dialect = engine.dialect.name
print(f"Database type: {dialect}")

# Check current email column
columns = inspector.get_columns('user')
email_col = next((col for col in columns if col['name'] == 'email'), None)
if email_col:
    print(f"Current email column: nullable={email_col.get('nullable', False)}")

with engine.connect() as conn:
    if dialect == 'sqlite':
        print("\n正在修改 SQLite 数据库...")

        # SQLite 需要重建表
        print("1. 创建新表...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_new (
                id TEXT PRIMARY KEY,
                email TEXT,
                username TEXT NOT NULL UNIQUE,
                role TEXT,
                name TEXT,
                profile_image_url TEXT,
                profile_banner_image_url TEXT,
                bio TEXT,
                gender TEXT,
                date_of_birth DATE,
                timezone TEXT,
                presence_state TEXT,
                status_emoji TEXT,
                status_message TEXT,
                status_expires_at BIGINT,
                info JSON,
                settings JSON,
                oauth JSON,
                last_active_at BIGINT,
                updated_at BIGINT,
                created_at BIGINT
            )
        """))

        print("2. 复制数据...")
        conn.execute(text("""
            INSERT INTO user_new
            SELECT * FROM user
        """))

        print("3. 删除旧表...")
        conn.execute(text("DROP TABLE user"))

        print("4. 重命名新表...")
        conn.execute(text("ALTER TABLE user_new RENAME TO user"))

        conn.commit()
        print("\n✓ SQLite 数据库修改完成!")

    elif dialect == 'postgresql':
        print("\n正在修改 PostgreSQL 数据库...")
        conn.execute(text("ALTER TABLE \"user\" ALTER COLUMN email DROP NOT NULL"))
        conn.commit()
        print("\n✓ PostgreSQL 数据库修改完成!")
    else:
        print(f"\n不支持的数据库类型: {dialect}")
        sys.exit(1)

print("\n现在 email 字段已经可以为 NULL 了!")
