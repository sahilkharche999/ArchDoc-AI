import os 
import base64
import json

from langgraph_temp_workflow.common.state import Sementic_Segmentation_State
from langgraph_temp_workflow.common.utils import image_to_data_url,crop_image_dynamic,pil_to_data_url,normalize_bbox

from langgraph_temp_workflow.common.schemas import DetectionOutput,EvaluationOutput,ExtractedContent


from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MAX_RETRIES_PER_CROP = 15   
llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0) 

def detect_regions_node(state: Sementic_Segmentation_State):
    """Phase 1: Semantic Discovery."""
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


def select_next_node(state: Sementic_Segmentation_State):
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


def evaluate_crop_node(state: Sementic_Segmentation_State):
    """Phase 3: The Semantic Eye."""
    current_bbox = state["current_bbox"]
    label = state["current_region_label"]
    img_path = Path(state["image_path"])
    image_folder = Path(state["output_dir"]) / img_path.stem
    image_folder.mkdir(parents=True, exist_ok=True)
    retry_count = state["current_retry_count"]
    
    # Generate current view
    cropped_pil = crop_image_dynamic(img_path, current_bbox)
    
    full_data = image_to_data_url(img_path)
    crop_data = pil_to_data_url(cropped_pil)
    
    # Prompt emphasizing "Semantic Completeness"
    prompt = f"""
    Context: We are cropping the region "{label}" (Attempt #{retry_count + 1}).
    
    Image 1: Full Sheet (Global Context).
    Image 2: Current Crop (Local View).

    TASK: Determine if Image 2 is a "Complete Semantic Unit".
    
    A Complete Unit MUST include ALL of the following:
    1. **The Main Content:** The primary Drawing, Table, or Detail itself.
    2. **The Title Block:** Look DOWN. Is the text describing this drawing (e.g., "PLAN VIEW", "SCALE 1:50", "LOOSE LINTEL SCHEDULE") fully visible?
    3. **No Cutoffs:** Are lines leaving the frame abruptly? Does a wall or table row get sliced in half?
    4. **Floating Annotations (CRITICAL):** Look at the white space around the main content.
       - Are there **Grid Bubbles** (Circles with numbers/letters) floating on the left/top?
       - Are there **Section Cuts** (Hexagons/Circles with arrows) floating on the right/bottom?
       - Are there **Dimension Strings** (Lines with numbers) floating outside the walls?
       - **RULE:** If these elements are visually associated with this drawing in Image 1, they MUST be included in Image 2.

    INSTRUCTIONS:
    - If the Title is cut off at the bottom: You MUST request `expand_bottom`.
    - If dimensions, grid bubbles, or floating symbols (like Hexagons) are cut off on the sides: Request `expand_left` or `expand_right`.
    - If the crop is too tight (touching the ink): Request a small expansion (0.05).
    - If it's perfect (includes content + title + all floating elements + small margin): Status = "approved".
    
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
        if retry_count >= MAX_RETRIES_PER_CROP:
            print("  ! MAX RETRIES REACHED. Forcing expansion and saving.")
            safe_bbox = normalize_bbox([
                current_bbox[0] - 0.05, current_bbox[1] - 0.05,
                current_bbox[2] + 0.05, current_bbox[3] + 0.05
            ])
            final_pil = crop_image_dynamic(img_path, safe_bbox)
            os.makedirs(state['output_dir'], exist_ok=True)
            safe_label = "".join(x for x in label if x.isalnum())
            filename = f"{safe_label}.png"
            filepath = image_folder / filename
            final_pil.save(filepath)
            
            return {
                "final_crops": [{"label": label, "bbox": safe_bbox, "file": str(filepath), "status": "forced_save"}],
                "current_retry_count": 999 
            }

        # --- NORMAL PROCESSING ---
        if response.status == "approved":
            print(f"  > APPROVED. Saving.")
            safe_label = "".join(x for x in label if x.isalnum())
            os.makedirs(state['output_dir'], exist_ok=True)
            filename = f"{safe_label}.png"
            filepath = image_folder / filename
            cropped_pil.save(filepath)
            
            return {
                "final_crops": [{"label": label, "bbox": current_bbox, "file": str(filepath), "status": "approved"}],
                "current_retry_count": 999
            }
        else:
            print(f"  > Adjusting: {response.feedback}")
            print(f"  > Expand: L:{response.expand_left}, T:{response.expand_top}, R:{response.expand_right}, B:{response.expand_bottom}")
            
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


def extract_content_node(state: Sementic_Segmentation_State):
    """
    Phase 4: Targeted Extraction using Pydantic.
    """
    print("\n--- PHASE 4: CONTENT EXTRACTION (STRUCTURED) ---")
    
    final_crops = state.get("final_crops", [])
    extracted_data = {}
    
    # Create the structured extractor
    extractor = llm.with_structured_output(ExtractedContent)
    
    for item in final_crops:
        label = item["label"]
        file_path = item["file"]
        
        # SKIP LOGIC
        if "PLAN" in label.upper() and "LEGEND" not in label.upper() and "NOTES" not in label.upper():
            print(f"Skipping Plan View: {label}")
            continue
            
        print(f"Extracting data from: {label}")
        
        try:
            with open(file_path, "rb") as img_file:
                b64_string = base64.b64encode(img_file.read()).decode('utf-8')
            
            prompt = f"""
            Analyze this image crop titled: **"{label}"**.
            
            ### TASK:
            Extract the content belonging to "{label}" into a structured format.
            Ignore neighboring text/drawings.
            
            ### RULES:
            1. If it is a **TABLE/SCHEDULE**: 
               - Use the first column (e.g., Mark/Symbol) as the Key.
               - The rest of the row data is the Value.
            2. If it is **NOTES**:
               - Use the Note Number (1, 2, 3) as the Key.
            3. If it is a **DETAIL/LEGEND**:
               - Use the Item Name as Key and Description/Specs as Value.
            """
            
            response = extractor.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{b64_string}"}
                ])
            ])
            
            # Save to State
            extracted_data[label] = response.content
            
            # Save to File
            json_path = file_path.replace(".png", ".json")
            with open(json_path, "w") as f:
                json.dump(response.content, f, indent=2)
                
        except Exception as e:
            print(f"Error extracting {label}: {e}")
            
    return {"extracted_data": extracted_data}
