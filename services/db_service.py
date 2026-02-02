import sqlite3
from pathlib import Path

DB_PATH = Path("db/app.db")

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS data_sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,            -- file name / table name
        source_type TEXT,     -- file / mysql / postgres / sqlite
        path TEXT,            -- 文件路径/ DB 连接串
        tag TEXT,             -- 文档标签
        upload_time DATETIME  -- 上传时间
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS analysis_tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        description TEXT
    )
    """)

    conn.commit()
    conn.close()
