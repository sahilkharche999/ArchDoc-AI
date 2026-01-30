import os 
import base64
import json

#importing the state
from langgraph_temp_workflow.common.state import Sementic_Segmentation_State

#util functions
from langgraph_temp_workflow.common.utils import image_to_data_url,crop_image_dynamic,pil_to_data_url,normalize_bbox

#importing Schemas
from langgraph_temp_workflow.common.schemas import DetectionOutput,EvaluationOutput,ExtractedContent

#importing prompt
from langgraph_temp_workflow.workflows.segmentation.prompt import prompt_for_detect_regions_node
from langgraph_temp_workflow.workflows.segmentation.prompt import prompt_for_evaluate_crop_node
from langgraph_temp_workflow.workflows.segmentation.prompt import prompt_for_extract_content_node

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

    prompt =prompt_for_detect_regions_node()

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
    prompt = prompt_for_evaluate_crop_node(label,retry_count)

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
            
            prompt = prompt_for_extract_content_node(label)
            
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
