@echo off
echo Initializing database...

cd /d "%~dp0"

REM Remove old database
if exist data\webui.db del /f data\webui.db

REM Create database with SQLite
sqlite3 data\webui.db < init_schema.sql

echo Database initialized!
pause
