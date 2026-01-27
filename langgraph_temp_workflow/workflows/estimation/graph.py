from langgraph.graph import StateGraph, START, END
from langgraph_temp_workflow.common.state import ProjectState
from langgraph_temp_workflow.workflows.estimation.nodes import node_classify_pages
from langgraph_temp_workflow.workflows.estimation.nodes import node_process_text_rules
from langgraph_temp_workflow.workflows.estimation.nodes import node_process_details
from langgraph_temp_workflow.workflows.estimation.nodes import node_process_plans
from langgraph_temp_workflow.workflows.estimation.nodes import node_agent_4_merger

workflow = StateGraph(ProjectState)
workflow.add_node("classify", node_classify_pages)
workflow.add_node("process_text", node_process_text_rules)
workflow.add_node("process_details", node_process_details)
workflow.add_node("process_plans", node_process_plans)
workflow.add_node("agent_4_merger", node_agent_4_merger)

workflow.add_edge(START, "classify")
workflow.add_edge("classify", "process_text")
workflow.add_edge("classify", "process_details")
workflow.add_edge("process_text", "process_plans")
workflow.add_edge("process_details", "process_plans")
workflow.add_edge("process_plans", "agent_4_merger")
workflow.add_edge("agent_4_merger", END)

main_workflow = workflow.compile()
