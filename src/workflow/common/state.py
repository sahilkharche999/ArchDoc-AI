import operator
from typing import TypedDict, List, Dict, Any, Optional,Annotated
from src.workflow.common.schemas import Region

class ProjectState(TypedDict):
    pdf_path: str
    output_dir: str
    page_map: Dict[int, str] 
    detail_library: Dict[str, Any] 
    general_rules: str 
    final_bill_of_materials: Dict # Output from Agent 4
    floor_plan_images:List[str]
