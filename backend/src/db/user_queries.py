from src.db.connection import get_conn, release_conn
from src.logger import setup_logger

logger = setup_logger(__name__)

def create_user(user_id: str, email: str, password_hash: str, created_at: str):
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (id, email, password_hash, created_at)
            VALUES (%s, %s, %s, %s)
        """, (user_id, email, password_hash, created_at))
        conn.commit()
        cursor.close()
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to create user | error={str(e)}")
        raise
    finally:
        release_conn(conn)

def get_user_by_email(email: str):
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        row = cursor.fetchone()
        cursor.close()
        return row
    except Exception as e:
        logger.error(f"Failed to get user by email | error={str(e)}")
        raise
    finally:
        release_conn(conn)


def get_user_by_id(user_id: str):
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))  # ← WHERE id not email
        row = cursor.fetchone()
        cursor.close()
        return row
    except Exception as e:
        logger.error(f"Failed to get user by id | error={str(e)}")
        raise
    finally:
        release_conn(conn)


def update_gemini_key(user_id: str, encrypted_key: str):
    conn=get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users SET gemini_api_key = %s WHERE id = %s
        """, (encrypted_key, user_id))
        conn.commit()
        cursor.close()
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to update gemini key | error={str(e)}")
        raise
    finally:
        release_conn(conn)
