from typing import TypedDict, List, Dict, Any


class ProjectState(TypedDict):
    pdf_path: str
    output_dir: str
    page_map: Dict[int, str]
    detail_library: Dict[str, Any]
    general_rules: str
    final_bill_of_materials: Dict
    floor_plan_images: List[str]
    detected_details: List[Dict[str, Any]]
