from src.db.connection import get_conn,release_conn
from src.logger import setup_logger

logger = setup_logger(__name__)

def init_users_table():
    logger.debug("Initializing users table")
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                gemini_api_key TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TEXT
            )
        """)
        conn.commit()
        cursor.close()
        logger.debug("Users table ready")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to initialize users table | error={str(e)}")
        raise
    finally:
        release_conn(conn)