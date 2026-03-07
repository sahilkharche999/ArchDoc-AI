import sqlite3

conn = sqlite3.connect("checkpoints.sqlite")
cursor = conn.cursor()

cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")

rows = cursor.fetchall()

for r in rows:
    print(r[0])