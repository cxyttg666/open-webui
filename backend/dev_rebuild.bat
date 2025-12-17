@echo off
set SKIP_ALEMBIC=true
set CORS_ALLOW_ORIGIN=http://localhost:5173;http://localhost:8080
if "%PORT%"=="" set PORT=8080

echo Creating database using SQLAlchemy...
python -c "import os; os.environ['SKIP_ALEMBIC']='true'; from open_webui.internal.db import Base, engine; Base.metadata.create_all(bind=engine); print('Database created')"

echo Starting server...
uvicorn open_webui.main:app --port %PORT% --host 0.0.0.0 --forwarded-allow-ips '*' --reload
