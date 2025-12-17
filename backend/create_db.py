#!/usr/bin/env python3
"""直接创建所有数据库表，不使用迁移"""
import sys
import os
from pathlib import Path

# 添加 backend 到路径
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

print("开始创建数据库...")

# 设置环境变量以避免运行迁移
os.environ['SKIP_ALEMBIC'] = 'true'

try:
    from open_webui.internal.db import Base, engine, get_db

    print("创建所有表...")
    Base.metadata.create_all(bind=engine)

    # 创建 alembic_version 表并设置版本
    from sqlalchemy import text
    with engine.connect() as conn:
        # 创建 alembic_version 表
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS alembic_version (
                version_num VARCHAR(32) NOT NULL,
                CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
            )
        """))

        # 设置为合并后的版本
        conn.execute(text("DELETE FROM alembic_version"))
        conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('merge_heads_20251215')"))
        conn.commit()

    # 验证表
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    print(f"\n成功创建 {len(tables)} 个表:")
    for table in sorted(tables):
        print(f"  ✓ {table}")

    print("\n数据库初始化完成！")

except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
