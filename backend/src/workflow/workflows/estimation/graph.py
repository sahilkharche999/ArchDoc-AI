from langgraph.graph import StateGraph, START, END
from src.workflow.common.state import ProjectState
from src.workflow.workflows.estimation.nodes import node_classify_pages
from src.workflow.workflows.estimation.nodes import node_process_text_rules
from src.workflow.workflows.estimation.nodes import node_process_details
from src.workflow.workflows.estimation.nodes import node_process_plans
from src.workflow.workflows.estimation.nodes import node_agent_4_merger

def route_after_hitl(state):
    remaining = state.get("remaining_pages", [])
    if len(remaining) > 0:
        return "process_plans"
    return "process_details"

def route_after_section_hitl(state):
    remaining = state.get("remaining_section_pages", [])
    if len(remaining) > 0:
        return "process_details"   # loop back for next section page
    return "agent_4_merger"
    
workflow = StateGraph(ProjectState)
workflow.add_node("classify", node_classify_pages)
workflow.add_node("process_text", node_process_text_rules)
workflow.add_node("process_plans", node_process_plans)
workflow.add_node("process_details", node_process_details)
workflow.add_node("agent_4_merger", node_agent_4_merger)

workflow.add_edge(START, "classify")
workflow.add_edge("classify", "process_text")
workflow.add_edge("process_text", "process_plans")
workflow.add_conditional_edges(
    "process_plans",
    route_after_hitl
)
workflow.add_conditional_edges(
    "process_details",
    route_after_section_hitl
)

workflow.add_edge("agent_4_merger", END)

main_workflow = workflow.compile()

