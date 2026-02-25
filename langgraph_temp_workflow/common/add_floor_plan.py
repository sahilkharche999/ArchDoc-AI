import sqlite3
import json
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph_temp_workflow.workflows.estimation.graph import workflow  

# 1. Setup
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)
app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "job_123"}}

# 2. Define the CORRECT Floor Plan Paths
manual_floor_plan = [
    "output_temp/floor_3/floor_3/vlm/images/c2071a8eb39ff6495f84a2cb170897bc62a795ef8b60ce9e337bd32f615e99dc.jpg",
    "output_temp/floor_4/floor_4/vlm/images/cb7ea89114e1c238311cf9bf3f1babcc1ef68eec3373691da3efe37289b125fe.jpg"
]

# 3. Inject State & Rewind
print("Injecting Manual Floor Plan and Rewinding to Agent 4...")

app.update_state(
    config, 
    {
        "floor_plan_images": manual_floor_plan # <--- FIXED: No extra brackets
    }, 
    as_node="process_details" 
)

# 4. Run Agent 4
print("Resuming execution...")
for event in app.stream(None, config=config):
    for node_name, state_update in event.items():
        print(f"Running Node: {node_name}")
        
        if node_name == "agent_4_merger":
            print("\n=== MERGER RESULT ===")
            print(json.dumps(state_update, indent=2))
            