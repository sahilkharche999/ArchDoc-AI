
from langgraph.graph import StateGraph, START, END
from langgraph_temp_workflow.common.state import Sementic_Segmentation_State
from langgraph_temp_workflow.workflows.segmentation.node import detect_regions_node
from langgraph_temp_workflow.workflows.segmentation.node import select_next_node
from langgraph_temp_workflow.workflows.segmentation.node import evaluate_crop_node

workflow = StateGraph(Sementic_Segmentation_State)
workflow.add_node("detect", detect_regions_node)
workflow.add_node("select_next", select_next_node)
workflow.add_node("evaluate", evaluate_crop_node)
# workflow.add_node("extract", extract_content_node) # Add new node

workflow.add_edge(START, "detect")
workflow.add_edge("detect", "select_next")

def queue_router(state):
    # If queue is empty, go to Extraction instead of END
    # return "extract" if state["current_region_label"] is None else "evaluate"
    return "END" if state["current_region_label"] is None else "evaluate"

# workflow.add_conditional_edges("select_next", queue_router, {"evaluate": "evaluate", "extract": "extract"})
workflow.add_conditional_edges("select_next", queue_router, {"evaluate": "evaluate", "END": END})

def eval_router(state):
    return "select_next" if state["current_retry_count"] >= 999 else "evaluate"

workflow.add_conditional_edges("evaluate", eval_router, {"evaluate": "evaluate", "select_next": "select_next"})

# workflow.add_edge("extract", END) # End after extraction

semantic_segmentation_app = workflow.compile()
