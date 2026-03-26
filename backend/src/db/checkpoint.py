import psycopg
from langgraph.checkpoint.postgres import PostgresSaver

from src.db.connection import pg_conn_string


def _init_memory() -> PostgresSaver:
    conn = psycopg.connect(pg_conn_string, autocommit=True)
    saver = PostgresSaver(conn)
    saver.setup()
    return saver


memory = _init_memory()