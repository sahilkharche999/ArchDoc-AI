from typing import TypedDict, List, Dict, Any,Optional

class PageState(TypedDict, total=False):
    page_num: int
    image_path: str
    pdf_path: str
    json_path: str
    sheet_number: str

    detected_bboxes: List[Dict[str, int]]
    corrected_bboxes: List[Dict[str, int]]

    status: str  


class SectionPageState(TypedDict, total=False):
    page_num: int
    image_path: str

    detected_bboxes: List[Dict[str, int]]
    corrected_bboxes: List[Dict[str, int]]

    status: str


class ProjectState(TypedDict):
    pdf_path: str
    output_dir: str

    page_map: Dict[int, str]

    detail_library: Dict[str, Any]
    general_rules: str
    final_bill_of_materials: Dict

    floor_plan_images: List[Dict[str, Any]]
    detected_details: List[Dict[str, Any]]

    remaining_pages: List[int]
    current_page: Optional[PageState]

    remaining_section_pages: List[int]
    current_section_page: Optional[SectionPageState]
    sheet_prefix: str
    temp_dependent_details: List[Dict[str, Any]]
    temp_plan_like_details: List[Dict[str, Any]]