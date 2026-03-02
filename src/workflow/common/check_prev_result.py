import sqlite3
import json
from langgraph.checkpoint.sqlite import SqliteSaver
from src.workflow.workflows.estimation.graph import workflow
from src.workflow.common.logger import setup_logger

logger = setup_logger(__name__)

# 1. Connect to the EXISTING database
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

# 2. Compile (Required to read the structure)
app = workflow.compile(checkpointer=memory)

# 3. Use the SAME Thread ID you used before
config = {"configurable": {"thread_id": "job_123"}}

# 4. Get the State directly (No API calls!)   
snapshot = app.get_state(config)

if not snapshot.values:
    logger.error("No data found for this thread ID.")
else:
    logger.info("\n=== PREVIOUS RUN RESULTS ===")
    
    # Print the Final Bill of Materials 
    bom = snapshot.values.get("detail_library", {})
    logger.info(json.dumps(bom, indent=2))
        