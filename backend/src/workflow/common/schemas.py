from typing import Literal, List, Optional, Dict, Any

from pydantic import BaseModel, Field


# AGENT 1: CLASSIFIER NODE 
class DrawingTypeResponse(BaseModel):
    drawing_type: Literal["text", "floor", "section"]

# Schema for Agent 3 (Detail Extraction)
class MaterialItem(BaseModel):
    item_name: str = Field(description="Exact text from drawing e.g. MC6x15.1")
    material_type: str = Field(description="Category: W, HSS, C, L, FB, ROD")
    piece_length_ft: float | None = Field(
        description="Length per piece in decimal feet IF explicitly stated in the callout. "
                    "Examples: 0'-3\" -> 0.25, 10\" -> 0.833, 1'-6\" -> 1.5. "
                    "Set to null if length is variable, 'SEE PLAN', or unknown."
    )
    qty_rule: str = Field(description="Logic: 'FIXED: [Count]' or 'VARIABLE: [Dependency]'")
    notes: Optional[str] = Field(description="Context notes e.g. 'Side Rails'")
    inherited_from: Optional[str] = Field(
        default=None,
        description="If this material came from a sub-detail, the detail ID e.g. '3/S3-01'"
    )

class FabricationMetrics(BaseModel):
    bolt_count: int = Field(default=0, description="Total bolts in this detail")
    hole_count: int = Field(default=0, description="Total holes (usually bolts * 1 or 2)")
    weld_inches: float = Field(default=0.0, description="Total linear inches of weld")

class DetailExtraction(BaseModel):
    detail_number: Optional[str] = Field(description="The number inside the bubble e.g. '7'")
    title: str = Field(description="Title of the detail e.g. 'LADDER DETAIL'")
    visual_reasoning: str = Field(description="CoT rationale: What is this drawing and how does it work?")
    materials: List[MaterialItem] = Field(description="List of ingredients")
    fabrication: FabricationMetrics = Field(description="Fabrication counts per detail instance")

class BillOfMaterialItem(BaseModel):
    description: str = Field(description="Human readable description e.g. 'Beam at Grid A'")
    material_size: str = Field(description="MUST match a value from the Valid Material List e.g. 'W24X62'")
 
    # The Core Metrics
    total_linear_feet: float = Field(description="Total length in feet")
    quantity: int = Field(description="Count of pieces")
 
    # Fabrication Metrics — all 3 required for Dax's 6-point takeoff
    total_bolts: int = Field(default=0, description="Total bolt count scaled by symbol occurrences on plan")
    total_holes: int = Field(default=0, description="Total hole count scaled by symbol occurrences on plan")
    total_weld_inches: float = Field(default=0.0, description="Total weld inches scaled by symbol occurrences on plan")
 
    # Pricing (populated post-LLM by enrich_bom_with_pricing)
    lb_per_ft: float | None = None
    total_weight_lbs: float | None = None
    charge_per_lb: float | None = None
 
    # The "Why" (CoT)
    logic_trace: str = Field(
        description="Explanation of the calculation. E.g. 'Found 5 cols. Height 18ft from Roof Note. 5*18=90ft.'")
 
    # Traceability — SPLIT into two fields for Dax review workflow
    source_drawing: str = Field(
        description="Combined reference kept for backward compatibility e.g. '2/S3-01'")
    source_sheet: str = Field(
        default="",
        description="The sheet number where this material was found e.g. 'S3-01'")
    source_symbol: str = Field(
        default="",
        description="The plan symbol or detail reference that triggered this material e.g. '2/S3-01' or 'hex-1'")
 
class FinalEstimation(BaseModel):
    project_summary: str = Field(
        description="2-3 sentence summary: what structure was estimated, how many sheets, key material categories found.")
    final_bill_of_materials: List[BillOfMaterialItem]

class IngestionOutput(BaseModel):
    type: Literal["Schedule", "Keyed_Notes", "Plan_View", "Detail", "Ignore"]
    title: Optional[str]
    columns: Optional[List[str]]
    rows: Optional[List[Dict[str, str]]]

class DetailGroup(BaseModel):
    detail_id: str = Field(description="The unique ID e.g. '7/S-3.2'")
    title: str = Field(description="The title text e.g. 'LADDER DETAIL'")
    image_files: List[str] = Field(description="List of image filenames belonging to this detail")
    text_blocks: List[str] = Field(description="List of text content belonging to this detail")

class DetailMap(BaseModel):
    groups: List[DetailGroup]
