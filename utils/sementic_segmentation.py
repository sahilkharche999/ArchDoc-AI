import operator
import os
import base64
from typing import TypedDict, Literal, List, Annotated, Optional
from PIL import Image
from io import BytesIO

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0) # 1.5 Pro is often sharper for OCR/Layout
MAX_RETRIES_PER_CROP = 5  # Reduced from 30 to prevent infinite loops

# --- SCHEMAS ---

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

# --- STATE ---

class GraphState(TypedDict):
    image_path: str
    detected_queue: List[Region]
    current_region_label: Optional[str]
    current_bbox: Optional[List[float]]
    current_retry_count: int
    final_crops: Annotated[List[dict], operator.add]

# --- HELPERS ---

def normalize_bbox(bbox: List[float]) -> List[float]:
    """Clamps coordinates between 0.0 and 1.0"""
    return [max(0.0, min(1.0, val)) for val in bbox]

def image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded_string}"

def pil_to_data_url(pil_image: Image.Image) -> str:
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded_string}"

def crop_image_dynamic(original_path: str, bbox: List[float]) -> Image.Image:
    img = Image.open(original_path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = bbox
    
    # Pixel conversion
    left, top = int(x1 * w), int(y1 * h)
    right, bottom = int(x2 * w), int(y2 * h)
    
    # Safety: Ensure box has area
    if right - left < 10 or bottom - top < 10:
        return img.crop((left, top, left+50, top+50))
        
    return img.crop((left, top, right, bottom))

# --- NODES ---

def detect_regions_node(state: GraphState):
    """
    Phase 1: Semantic Discovery.
    Finds the general areas, aiming for LOOSE bounding boxes.
    """
    print("\n--- PHASE 1: SEMANTIC DISCOVERY ---")
    img_path = state["image_path"]
    image_data = image_to_data_url(img_path)

    prompt = """
    You are a Senior Architectural Technologist.
    Analyze this sheet to identify **Logical Content Blocks**.

    A "Logical Block" includes:
    1. The graphical content (The drawing, detail, or table).
    2. **The Metadata**: The TITLE text, SCALE, and DETAIL NUMBER usually located BELOW the drawing.
    3. **The Gutter**: The empty whitespace immediately surrounding the content that separates it from neighbors.

    Goal: Return bounding boxes that loosely encapsulate these full blocks. 
    It is better to include slightly too much whitespace than to cut off a title.
    """

    detector = llm.with_structured_output(DetectionOutput)
    
    try:
        response = detector.invoke([
            SystemMessage(content="You are a layout analysis expert."),
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_data} 
            ])
        ])
        print(f"Detected {len(response.regions)} regions.")
        return {"detected_queue": response.regions}
    except Exception as e:
        print(f"Detection failed: {e}")
        return {"detected_queue": []}


def select_next_node(state: GraphState):
    """Phase 2: Queue Management."""
    queue = state.get("detected_queue", [])
    if not queue:
        return {"current_region_label": None, "current_bbox": None}
    
    next_item = queue[0]
    print(f"\n--- OPTIMIZING: {next_item.label} ---")
    
    return {
        "detected_queue": queue[1:],
        "current_region_label": next_item.label,
        "current_bbox": next_item.bbox,
        "current_retry_count": 0
    }


def evaluate_crop_node(state: GraphState):
    """
    Phase 3: The Semantic Eye. 
    Ensures the crop captures the full meaning (Drawing + Title).
    """
    current_bbox = state["current_bbox"]
    label = state["current_region_label"]
    img_path = state["image_path"]
    retry_count = state["current_retry_count"]
    
    # Generate current view
    cropped_pil = crop_image_dynamic(img_path, current_bbox)
    
    full_data = image_to_data_url(img_path)
    crop_data = pil_to_data_url(cropped_pil)
    
    # Prompt emphasizing "Semantic Completeness"
    prompt = f"""
    Context: We are cropping the region "{label}" (Attempt #{retry_count + 1}).
    
    Image 1: Full Sheet.
    Image 2: Current Crop.

    TASK: Determine if Image 2 is a "Complete Semantic Unit".
    
    A Complete Unit MUST include:
    1. The Main Content (Drawing/Table).
    2. **The Title Block**: Look DOWN. Is the text describing this drawing (e.g., "PLAN VIEW", "SCALE 1:50") visible?
    3. **No Cutoffs**: Are lines leaving the frame abruptly?

    INSTRUCTIONS:
    - If the Title is cut off at the bottom: You MUST request `expand_bottom`.
    - If dimensions are cut off on sides: Request `expand_left` or `expand_right`.
    - If it's perfect (includes content + title + small margin): Status = "approved".
    
    BE BOLD: If content is missing, expand by at least 0.05 or 0.10. Do not do tiny adjustments.
    """

    evaluator = llm.with_structured_output(EvaluationOutput)
    
    try:
        response = evaluator.invoke([
            SystemMessage(content="You are a strict Layout QA Auditor."),
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": full_data}, 
                {"type": "image_url", "image_url": crop_data}
            ])
        ])
        
        # --- FAILURE SAFEGUARD ---
        # If we have tried 5 times and still failing, force a save with extra padding
        if retry_count >= MAX_RETRIES_PER_CROP:
            print("  ! MAX RETRIES REACHED. Forcing expansion and saving.")
            # Add 5% padding as a safety measure and save
            safe_bbox = normalize_bbox([
                current_bbox[0] - 0.05, current_bbox[1] - 0.05,
                current_bbox[2] + 0.05, current_bbox[3] + 0.05
            ])
            final_pil = crop_image_dynamic(img_path, safe_bbox)
            
            safe_label = "".join(x for x in label if x.isalnum())
            filename = f"final_{safe_label}.png"
            final_pil.save(filename)
            
            return {
                "final_crops": [{"label": label, "bbox": safe_bbox, "file": filename, "status": "forced_save"}],
                "current_retry_count": 999 
            }

        # --- NORMAL PROCESSING ---
        if response.status == "approved":
            print(f"  > APPROVED. Saving.")
            safe_label = "".join(x for x in label if x.isalnum())
            filename = f"final_{safe_label}.png"
            cropped_pil.save(filename)
            
            return {
                "final_crops": [{"label": label, "bbox": current_bbox, "file": filename, "status": "approved"}],
                "current_retry_count": 999
            }
        else:
            print(f"  > Adjusting: {response.feedback}")
            print(f"  > Expand: L:{response.expand_left}, T:{response.expand_top}, R:{response.expand_right}, B:{response.expand_bottom}")
            
            # Apply Expansion Deltas
            # x1 moves left (minus), y1 moves up (minus), x2 moves right (plus), y2 moves down (plus)
            new_bbox = [
                current_bbox[0] - response.expand_left,
                current_bbox[1] - response.expand_top,
                current_bbox[2] + response.expand_right,
                current_bbox[3] + response.expand_bottom
            ]
            
            return {
                "current_bbox": normalize_bbox(new_bbox),
                "current_retry_count": retry_count + 1
            }

    except Exception as e:
        print(f"Error in evaluation: {e}")
        return {"current_retry_count": 999}

# --- GRAPH ---

workflow = StateGraph(GraphState)
workflow.add_node("detect", detect_regions_node)
workflow.add_node("select_next", select_next_node)
workflow.add_node("evaluate", evaluate_crop_node)

workflow.add_edge(START, "detect")
workflow.add_edge("detect", "select_next")

def queue_router(state):
    return "END" if state["current_region_label"] is None else "evaluate"

workflow.add_conditional_edges("select_next", queue_router, {"evaluate": "evaluate", "END": END})

def eval_router(state):
    return "select_next" if state["current_retry_count"] >= 999 else "evaluate"

workflow.add_conditional_edges("evaluate", eval_router, {"evaluate": "evaluate", "select_next": "select_next"})

app = workflow.compile()

# --- RUN ---

if __name__ == "__main__":
    input_image = "floor_1.png" 
    
    if os.path.exists(input_image):
        print(f"Processing {input_image}...")
        initial = {
            "image_path": input_image, 
            "detected_queue": [], 
            "final_crops": [], 
            "current_retry_count": 0,
            "current_region_label": None, 
            "current_bbox": None
        }
        
        # Increase recursion limit to handle the list iteration
        app.invoke(initial, config={"recursion_limit": 150})
    else:
        print("Image not found.")