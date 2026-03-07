import sqlite3

conn = sqlite3.connect("checkpoints.sqlite")
cursor = conn.cursor()

cursor.execute("""
INSERT OR REPLACE INTO jobs (job_id, name, file_name, status, upload_date)
VALUES (?, ?, ?, ?, ?)
""", (
    "job_123",
    "Hippo Vet Highland",
    "extracted_pages.pdf",
    "Completed",
    "Mar 6, 2026"
))

conn.commit()
conn.close()

print("Test job inserted")