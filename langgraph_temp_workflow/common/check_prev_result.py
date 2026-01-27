import sqlite3
import json
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph_temp_workflow.workflows.estimation.graph import workflow  

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
    print("No data found for this thread ID.")
else:
    print("\n=== PREVIOUS RUN RESULTS ===")
    
    # Print the Final Bill of Materials
    bom = snapshot.values.get("final_bill_of_materials", {})
    print(json.dumps(bom, indent=2))
    
    # Optional: Check other data
    # print(snapshot.values.get("detail_library", {}).keys())