from langgraph.checkpoint.postgres import PostgresSaver

from src.db.connection import pg_conn_string

memory = PostgresSaver.from_conn_string(pg_conn_string)
memory.setup()