from pydantic import BaseModel, Field
from typing import  Literal, List, Optional, Dict, Any

# ---SEMENTIC SEGMENTATION SCHEMAS START---
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
    
    # We ask for explicit movements
    expand_left: float = Field(description="Amount to move LEFT edge to the LEFT (0.0 to 1.0)", default=0.0)
    expand_top: float = Field(description="Amount to move TOP edge UP (0.0 to 1.0)", default=0.0)
    expand_right: float = Field(description="Amount to move RIGHT edge to the RIGHT (0.0 to 1.0)", default=0.0)
    expand_bottom: float = Field(description="Amount to move BOTTOM edge DOWN (0.0 to 1.0)", default=0.0)

class ExtractedContent(BaseModel):
    title: str = Field(description="The exact title of the section being extracted")
    category: Literal["Table", "Notes", "Detail", "Legend"] = Field(description="The type of content found")
    content: Dict[str, Any] = Field(description="The extracted data as a Key-Value dictionary. For tables, keys are row headers. For notes, keys are numbers.")

# ---SEMENTIC SEGMENTATION SCHEMAS END ---



# ---AGENTS SCHEMAS START---

# ---  AGENT 1: CLASSIFIER NODE ---
class DrawingTypeResponse(BaseModel):
    drawing_type: Literal["text", "floor", "section"]


# Schema for Agent 3 (Detail Extraction)
class MaterialItem(BaseModel):
    item_name: str = Field(description="Name of the material e.g. MC6x15.1")
    qty_rule: str = Field(description="How to calculate qty e.g. 'Count' or 'Length of Rail'")
    notes: Optional[str] = Field(description="Context notes")

class DetailExtraction(BaseModel):
    detail_number: Optional[str] = Field(description="The number inside the bubble e.g. '7'")
    title: str = Field(description="Title of the detail")
    materials: List[MaterialItem] = Field(description="List of BOM items found")

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
    name: str
    data: str

class PlanExtraction(BaseModel):
    sheet_type: str = Field(description="Floor Plan or Roof Plan")
    members: List[PlanMember]
    symbols: List[PlanSymbol]
    global_notes: List[str]
    schedules: List[PlanSchedule]

# Schema for Agent 4 (Final Merger)
class BillOfMaterialItem(BaseModel):
    description: str
    total_qty: float
    total_linear_feet: Optional[float]
    logic_trace: str

class FinalEstimation(BaseModel):
    final_bill_of_materials: List[BillOfMaterialItem]

