#!/usr/bin/env python3
"""Initialize database with all tables"""
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from open_webui.internal.db import Base, engine

print("Creating all database tables...")
Base.metadata.create_all(bind=engine)
print("Database initialized successfully!")

# Verify tables were created
from sqlalchemy import inspect
inspector = inspect(engine)
tables = inspector.get_table_names()
print(f"\nCreated {len(tables)} tables:")
for table in sorted(tables):
    print(f"  - {table}")
