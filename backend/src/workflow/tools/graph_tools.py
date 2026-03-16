from langchain_core.tools import tool
from src.infrastructure import graph_db
from src.workflow.common.schemas import FinalEstimation

@tool
def lookup_symbol_definition(symbol_description: str, project_id: str):
    """
    Use this tool to find the definition of a symbol found on a plan.
    Input: 
      - symbol_description: A visual description or text label (e.g., "Hexagon 1", "7/S-3.2").
      - project_id: The ID of the current project (filename).
    Output: The material specs or schedule rules.
    """
    # Try semantic search 
    matches = graph_db.semantic_search(symbol_description, project_id, limit=1)
    
    if matches and matches[0]['score'] > 0.85:
        match = matches[0]
        return {
            "found": True,
            "type": "Detail" if match.get("BOM") else "Rule",
            "specs": match.get("Specs"),
            "bom": match.get("BOM")
        }
    return {"found": False, "message": "No definition found."}


@tool
def submit_final_estimate(estimation: FinalEstimation):
    """
    Call this tool when you have finished calculating the Bill of Materials.
    Pass the final JSON object here.
    """
    return estimation # This returns the Pydantic object directly