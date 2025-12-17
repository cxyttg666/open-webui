#!/usr/bin/env python3
"""Fix auth table to add username column"""
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from open_webui.internal.db import engine
from sqlalchemy import text, inspect

print("检查 auth 表结构...")
inspector = inspect(engine)

try:
    columns = inspector.get_columns('auth')
    print("\n当前 auth 表结构:")
    for col in columns:
        print(f"  - {col['name']}: {col['type']} (nullable={col.get('nullable', 'unknown')})")

    # 检查是否有 username 列
    has_username = any(col['name'] == 'username' for col in columns)

    if not has_username:
        print("\n缺少 username 列，正在添加...")

        with engine.connect() as conn:
            # 检查是否有 email 列
            has_email = any(col['name'] == 'email' for col in columns)

            if has_email:
                # 如果有 email 列，将其重命名为 username
                print("将 email 列重命名为 username...")
                # SQLite 需要重建表
                conn.execute(text("""
                    CREATE TABLE auth_new (
                        id TEXT PRIMARY KEY,
                        username TEXT UNIQUE,
                        password TEXT,
                        active BOOLEAN
                    )
                """))

                conn.execute(text("""
                    INSERT INTO auth_new (id, username, password, active)
                    SELECT id, email, password, active FROM auth
                """))

                conn.execute(text("DROP TABLE auth"))
                conn.execute(text("ALTER TABLE auth_new RENAME TO auth"))
            else:
                # 直接添加 username 列
                print("添加 username 列...")
                conn.execute(text("""
                    ALTER TABLE auth ADD COLUMN username TEXT UNIQUE
                """))

            conn.commit()
            print("✓ auth 表修复完成!")
    else:
        print("\n✓ auth 表已经有 username 列")

except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n验证修改...")
columns = inspector.get_columns('auth')
print("修改后的 auth 表结构:")
for col in columns:
    print(f"  - {col['name']}: {col['type']} (nullable={col.get('nullable', 'unknown')})")
