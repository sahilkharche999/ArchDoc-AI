from typing import Literal, List, Optional, Dict, Any

from pydantic import BaseModel, Field


class Region(BaseModel):
    id: int = Field(description="Unique identifier")
    label: str = Field(description="Description (e.g., 'Main Floor Plan', 'Door Schedule')")
    bbox: List[float] = Field(description="Normalized coordinates [x1, y1, x2, y2] (0.0 to 1.0)")


class DetectionOutput(BaseModel):
    regions: List[Region] = Field(description="List of all detected logical regions")


class EvaluationOutput(BaseModel):
    status: Literal["approved", "needs_adjustment"] = Field(
        description="Approved ONLY if the crop contains the Drawing, The Title, AND the whitespace gutter around it."
    )
    feedback: str = Field(description="Specific description of what is missing or if there is too much noise.")

    expand_left: float = Field(description="Amount to move LEFT edge to the LEFT (0.0 to 1.0)", default=0.0)
    expand_top: float = Field(description="Amount to move TOP edge UP (0.0 to 1.0)", default=0.0)
    expand_right: float = Field(description="Amount to move RIGHT edge to the RIGHT (0.0 to 1.0)", default=0.0)
    expand_bottom: float = Field(description="Amount to move BOTTOM edge DOWN (0.0 to 1.0)", default=0.0)


class ExtractedContent(BaseModel):
    title: str = Field(description="The exact title of the section being extracted")
    category: Literal["Table", "Notes", "Detail", "Legend"] = Field(description="The type of content found")
    content: Dict[str, Any] = Field(
        description="The extracted data as a Key-Value dictionary. For tables, keys are row headers. For notes, keys are numbers.")


# ---  AGENT 1: CLASSIFIER NODE ---
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


# The Recipe Card (The Main Object)
class DetailExtraction(BaseModel):
    detail_number: Optional[str] = Field(description="The number inside the bubble e.g. '7'")
    title: str = Field(description="Title of the detail e.g. 'LADDER DETAIL'")
    visual_reasoning: str = Field(description="CoT rationale: What is this drawing and how does it work?")
    materials: List[MaterialItem] = Field(description="List of ingredients")
    fabrication: FabricationMetrics = Field(description="Fabrication counts per detail instance")


class DetailList(BaseModel):
    details: List[DetailExtraction]


# Schema for Agent 2 (Plan Extraction)
class PlanMember(BaseModel):
    label: str = Field(description="Text label e.g. W24x62")
    location: str = Field(description="Grid location e.g. B-2")
    length_text: Optional[str] = Field(description="Dimension text found nearby e.g. 27'-0\"")
    count: int = Field(default=1)


class PlanSymbol(BaseModel):
    symbol: str = Field(description="The text inside the symbol e.g. 7/S-3.2")
    location: str = Field(description="Grid location")
    associated_text: Optional[str] = Field(description="Dimension text found nearby e.g. 13'-10\"")


class PlanSchedule(BaseModel):
    name: str = Field(description="Title of the schedule e.g. 'Shear Wall Schedule'")
    data: str = Field(description="The content of the schedule as a string or JSON-like string")


class PlanExtraction(BaseModel):
    # NEW FIELD: Tells us if this crop is a Drawing or a Definition Table
    content_type: Literal["Plan_View", "Definition_Schedule", "Notes"] = Field(
        description="Classify the image crop: 'Plan_View' for drawings, 'Definition_Schedule' for tables/legends."
    )
    visual_reasoning: str = Field(description="CoT rationale: What did you see and how did you classify it?")

    # Fields for Plan View (Instances)
    members: List[PlanMember] = Field(default=[])
    symbols: List[PlanSymbol] = Field(default=[])

    # Fields for Schedules/Notes (Definitions)
    schedules: List[PlanSchedule] = Field(default=[])
    global_notes: List[str] = Field(default=[])



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


class ScheduleRule(BaseModel):
    schedule_name: str = Field(description="Name of the schedule e.g. 'Shear Wall Schedule'")
    symbol: str = Field(description="The symbol being defined e.g. '<1>' or 'F5'")
    specs: str = Field(description="The definition e.g. '5/8 bolt @ 16oc'")

class Rule(BaseModel):
    rule_number: int
    text: str

class Section(BaseModel):
    section_name: str
    rules: List[Rule]

class TextRulesExtraction(BaseModel):
    sections: List[Section]
    general_notes: List[str]



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

class BOMValidationItem(BaseModel):
    index: int = Field(description="Index of the item in the input list (0-based)")
    action: Literal["keep", "drop"] = Field(description="keep or drop this item")
    reason: str = Field(description="Why keeping or dropping")

class BOMValidation(BaseModel):
    validated_items: List[BOMValidationItem]