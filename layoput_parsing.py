import os
import json
import base64
from PIL import Image
from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# --- SCHEMAS ---
class MergedBlock(BaseModel):
    title: str = Field(description="The inferred title of this block")
    type: str = Field(description="Schedule, Notes, or Plan")
    # The VLM will calculate the union of the bboxes
    final_bbox: List[int] = Field(description="[x1, y1, x2, y2] covering the Title AND the Content")
    component_indices: List[int] = Field(description="Indices of the JSON items included in this block")

class LayoutAnalysis(BaseModel):
    blocks: List[MergedBlock]

# --- HELPERS ---
def load_image_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def assemble_layout(image_path, json_path):
    print(f"Assembling Layout for {image_path}...")
    
    # 1. Load Image & JSON
    img = Image.open(image_path)
    w, h = img.size
    
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
        if isinstance(raw_data, list) and len(raw_data) > 0 and isinstance(raw_data[0], list):
            raw_data = raw_data[0] # Flatten if nested list

    # 2. Simplify JSON for Prompt (Save Tokens)
    # We only send Index, Type, Text snippet, and BBox
    simplified_items = []
    for i, item in enumerate(raw_data):
        text_snippet = ""
        if item.get("type") == "title":
            text_snippet = item["content"]["title_content"][0]["content"]
        elif item.get("type") == "text":
            text_snippet = item.get("text", "")[:50]
        
        simplified_items.append({
            "id": i,
            "type": item.get("type"),
            "bbox": item.get("bbox"), # MinerU bbox (PDF coords usually)
            "text_preview": text_snippet
        })
    
    json_str = json.dumps(simplified_items, indent=2)

    # 3. The "Assembler" Prompt
    prompt = f"""
    You are a Layout Analysis Engine.
    I have parsed a PDF into a list of disjointed items (Titles, Tables, Lists, Images).
    
    ### INPUT DATA:
    1. **Image:** The visual layout of the page.
    2. **JSON List:** The detected items with their coordinates.
    
    ### YOUR TASK:
    Group these items into **Logical Blocks**.
    - A "Block" usually consists of a **Title** (e.g. "SHEET KEYED NOTES") and the **Content** below it (List/Table).
    - Sometimes a Block is just a large Image (Plan View).
    
    ### INSTRUCTIONS:
    1. Look at the Image to see which items are visually grouped.
    2. Look at the JSON 'bbox' to confirm they are close to each other.
    3. Create a `MergedBlock` for each group.
    4. Calculate the `final_bbox` that encompasses ALL items in that group.
       - **CRITICAL:** The `final_bbox` must be in the same coordinate system as the input JSON (PDF coords).
    
    ### RAW JSON ITEMS:
    {json_str}
    """

    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": f"data:image/png;base64,{load_image_base64(img)}"}
    ])

    # 4. Invoke VLM
    try:
        result = llm.with_structured_output(LayoutAnalysis).invoke([msg])
        
        print(f"Found {len(result.blocks)} logical blocks.")
        
        # 5. Process Results (Scaling & Cropping)
        # Note: MinerU bboxes are usually 1000x1000 or PDF points. 
        # We need to scale them to the Image Size (w, h).
        # We assume the VLM returns the bbox in the SAME scale as the input JSON.
        
        # We need to know the PDF dimensions to scale correctly.
        # Assuming standard MinerU 1000x1000 normalization if not specified, 
        # BUT your JSON shows coords like [740, 26]. This looks like PDF points (72dpi).
        # Let's assume the VLM returns what it saw in the JSON.
        
        # To crop the image (which might be 300dpi), we need a scale factor.
        # Heuristic: Compare Image Width to Max JSON X.
        max_json_x = max([item["bbox"][2] for item in simplified_items if item.get("bbox")])
        scale_factor = w / max_json_x
        
        output_crops = []
        
        for block in result.blocks:
            # Scale the bbox
            x1, y1, x2, y2 = block.final_bbox
            
            # Add padding
            pad = 20
            crop_box = (
                int(x1 * scale_factor) - pad,
                int(y1 * scale_factor) - pad,
                int(x2 * scale_factor) + pad,
                int(y2 * scale_factor) + pad
            )
            
            # Clamp
            crop_box = (
                max(0, crop_box[0]), max(0, crop_box[1]),
                min(w, crop_box[2]), min(h, crop_box[3])
            )
            
            # Crop
            crop_img = img.crop(crop_box)
            
            # Save
            safe_title = "".join(x for x in block.title if x.isalnum())
            save_path = f"output_temp/merged_{safe_title}.png"
            crop_img.save(save_path)
            
            output_crops.append({
                "title": block.title,
                "type": block.type,
                "path": save_path
            })
            print(f"  > Saved Merged Crop: {save_path}")
            
        return output_crops

    except Exception as e:
        print(f"Assembler Failed: {e}")
        return []

if __name__ == "__main__":
    # Test it
    assemble_layout("S101FLOOR.png", "output/Foundation_plan/vlm/Foundation_plan_content_list_v2.json")