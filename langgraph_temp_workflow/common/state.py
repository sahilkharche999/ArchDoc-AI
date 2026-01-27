import operator
from typing import TypedDict, List, Dict, Any, Optional,Annotated
from langgraph_temp_workflow.common.schemas import Region

class ProjectState(TypedDict):
    pdf_path: str
    output_dir: str
    page_map: Dict[int, str] 
    detail_library: Dict[str, Any] 
    general_rules: str 
    raw_plan_data: List[Dict] # Output from Agent 2
    final_bill_of_materials: Dict # Output from Agent 4
    
class Sementic_Segmentation_State(TypedDict):
    image_path: str
    detected_queue: List[Region]
    current_region_label: Optional[str]
    current_bbox: Optional[List[float]]
    current_retry_count: int
    final_crops: Annotated[List[dict], operator.add]
    output_dir: str 
    extracted_data: Dict[str, Any] # New field for final JSON data 

