import cv2
import re 
import pdfplumber
import json
import os
import base64
from typing import  List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from io import BytesIO
from google import genai
import pandas as pd
from langgraph_temp_workflow.common.schemas import DetailExtraction
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_path  # NEW IMPORT
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List, Optional
from langgraph_temp_workflow.workflows.estimation.prompt import prompt_for_map_page_layout
from langgraph_temp_workflow.workflows.estimation.prompt import prompt_for_extract_single_detail
load_dotenv()

# --- 1. SETUP MODELS ---
llm_pro = ChatGoogleGenerativeAI(model="gemini-3-pro-preview") 
llm_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash") 
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL = "gemini-2.5-flash" 

def crop_union_tables(json_path, image_path, output_dir="debug_crops"):
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    full_img = Image.open(image_path)
    img_w, img_h = full_img.size
    print(f"Loaded Image: {img_w}x{img_h}")

    with open(json_path, 'r') as f:
        content_list = json.load(f)
        # Handle nested list structure [[...]]
        if isinstance(content_list, list) and len(content_list) > 0 and isinstance(content_list[0], list):
            content_list = content_list[0]

    # --- CALCULATE SCALE FACTOR ---
    max_json_x = 0
    max_json_y = 0
    for item in content_list:
        if item.get("bbox"):
            max_json_x = max(max_json_x, item["bbox"][2])
            max_json_y = max(max_json_y, item["bbox"][3])
    
    if max_json_x == 0: max_json_x = 1000 
    scale_x = img_w / max_json_x
    scale_y = img_h / max_json_y
    
    print(f"Detected Scale Factor: X={scale_x:.2f}, Y={scale_y:.2f}")

    processed_indices = set()

    # --- ITERATE TO FIND TITLES ---
    for i, item in enumerate(content_list):
        if i in processed_indices: continue

        item_type = item.get("type")
        bbox = item.get("bbox") 
        
        if not bbox: continue

        # 1. Is this a Title?
        if item_type == "title":
            # Extract text
            try:
                title_text = item["content"]["title_content"][0]["content"]
                safe_title = "".join(x for x in title_text if x.isalnum() or x == " ")[:30].strip()
            except:
                safe_title = f"Title_{i}"

            print(f"Checking Title: '{safe_title}'...")

            # 2. Search for the Body (Spatial Search)
            best_match_idx = -1
            min_gap = 1000

            for j, candidate in enumerate(content_list):
                if i == j or j in processed_indices: continue
                
                cand_type = candidate.get("type")
                cand_bbox = candidate.get("bbox")
                
                if not cand_bbox: continue

                # We look for Tables or Lists
                if cand_type in ["table", "list"]:
                    
                    # Check Vertical Gap (Candidate must be BELOW Title)
                    gap = cand_bbox[1] - bbox[3]
                    
                    # Check Horizontal Alignment (Must overlap in X)
                    # Overlap = max(0, min(r1, r2) - max(l1, l2))
                    overlap_x = max(0, min(bbox[2], cand_bbox[2]) - max(bbox[0], cand_bbox[0]))
                    
                    # Rules:
                    # 1. Must be below (gap > -10 to allow slight overlap)
                    # 2. Must be close (gap < 100)
                    # 3. Must align horizontally (overlap > 0)
                    if -10 < gap < 100 and overlap_x > 0:
                        if gap < min_gap:
                            min_gap = gap
                            best_match_idx = j

            # 3. If Match Found -> Union Crop
            if best_match_idx != -1:
                body_item = content_list[best_match_idx]
                body_bbox = body_item["bbox"]
                print(f"  -> MATCH! Found '{body_item['type']}' below (Gap: {min_gap:.1f})")

                # Calculate Union Box
                union_x1 = min(bbox[0], body_bbox[0]) - 60 # Padding for Symbol
                union_y1 = bbox[1] - 10
                union_x2 = max(bbox[2], body_bbox[2]) + 10
                union_y2 = body_bbox[3] + 10
                
                # Scale
                crop_box = (
                    int(union_x1 * scale_x),
                    int(union_y1 * scale_y),
                    int(union_x2 * scale_x),
                    int(union_y2 * scale_y)
                )
                
                # Clamp
                crop_box = (
                    max(0, crop_box[0]), max(0, crop_box[1]),
                    min(img_w, crop_box[2]), min(img_h, crop_box[3])
                )

                # Crop & Save
                try:
                    crop_img = full_img.crop(crop_box)
                    save_path = os.path.join(output_dir, f"UNION_{safe_title}.png")
                    crop_img.save(save_path)
                    print(f"  -> Saved Union Crop: {save_path}")
                    
                    # Mark both as processed
                    processed_indices.add(i)
                    processed_indices.add(best_match_idx)
                    
                    # OPTIONAL: Delete the old MinerU image if it exists
                    try:
                        old_path = body_item["content"]["image_source"]["path"]
                        # Construct full path and delete...
                    except: pass

                except Exception as e:
                    print(f"  ! Crop Failed: {e}")

        # 4. If it's a Table/List that wasn't merged (Orphan)
        elif item_type in ["table", "list"] and i not in processed_indices:
            # Just use the existing image if available, or crop it fresh
            # This handles tables that MinerU found perfectly without a separate title
            pass 

# --- HELPER ---
def image_to_base64(image_obj):
    buff = BytesIO()
    image_obj.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

class DetailGroup(BaseModel):
    detail_id: str = Field(description="The unique ID e.g. '7/S-3.2'")
    title: str = Field(description="The title text e.g. 'LADDER DETAIL'")
    image_files: List[str] = Field(description="List of image filenames belonging to this detail")
    text_blocks: List[str] = Field(description="List of text content belonging to this detail")

class DetailMap(BaseModel):
    groups: List[DetailGroup]

# --- STEP 1: THE MAPPER ---
def map_page_layout(pdf_layout_path: str, json_path: str, images_dir: str):
    """
    Uses VLM to look at the full page layout and group items into 'Detail Units'.
    Returns a list of DetailGroup objects.
    """
    print(f"   > Mapping Layout for {pdf_layout_path}...")
    
    # 1. Load Context
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        # Simplify JSON for prompt (just types and bboxes)
        simple_json = [{"id": i, "type": x["type"], "bbox": x.get("bbox"), "text_preview": x.get("text", "")[:50]} for i, x in enumerate(json_data)]
        json_string = json.dumps(simple_json, indent=2)

    # 2. Convert Layout PDF to Image
    try:
        layout_images = convert_from_path(pdf_layout_path)
        layout_image_b64 = image_to_base64(layout_images[0])
    except:
        return []

    # 3. Prompt
    prompt = prompt_for_map_page_layout()
    
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{layout_image_b64}"}},
        {"type": "text", "text": f"JSON Items:\n{json_string}"}
    ])
    
    try:
        # Use Flash for layout mapping (it's fast and good at spatial grouping)
        result = llm_flash.with_structured_output(DetailMap).invoke([msg])
        print(f"This is how the group looks like : {result.groups}")
        return result.groups
    except Exception as e:
        print(f"   ! Mapping failed: {e}")
        return []

# --- STEP 2: THE EXTRACTOR ---
def extract_single_detail(group: DetailGroup, images_dir: str):
    """
    Analyzes a SINGLE detail group (specific images + text) to get the BOM.
    """
    print(f"   > Extracting BOM for {group.detail_id}...")
    
    payload = []
    
    # A. Prompt
    
    prompt =prompt_for_extract_single_detail(group.title,group.detail_id)

    payload.append({"type": "text", "text": prompt})
    
    # B. Add Specific Images
    for img_file in group.image_files:
        # Handle full path or relative path from JSON
        fname = os.path.basename(img_file)
        full_path = os.path.join(images_dir, fname)
        if os.path.exists(full_path):
            b64 = load_image_base64(Image.open(full_path))
            payload.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    
    # C. Add Specific Text
    if group.text_blocks:
        payload.append({"type": "text", "text": "NOTES:\n" + "\n".join(group.text_blocks)})
        
    try:
        # Use Pro for reading the engineering text
        result = llm_pro.with_structured_output(DetailExtraction).invoke([HumanMessage(content=payload)])
        return result
    except Exception as e:
        print(f"   ! Extraction failed for {group.detail_id}: {e}")
        return None

def get_valid_materials_list(excel_path):
    try:
        df = pd.read_excel(excel_path, sheet_name="Options")
        return df.iloc[:, 0].dropna().astype(str).tolist()
    except:
        return [] 
    
def load_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def filter_candidate_coordinates(image_path, all_candidates_flat_list):
    """
    Filter candidates using :
    1. Geometric Subset Removal (Box inside Box)
    2. VLM Visual Inspection (Final tie-breaker)
    """
    print("--- Filtering Noise Coordinates ---")

    # 1. Geometric Subset Removal
    for item in all_candidates_flat_list:
        c = item["coords"]
        item["area"] = (c["x2"] - c["x1"]) * (c["y2"] - c["y1"])

    sorted_candidates = sorted(all_candidates_flat_list, key=lambda x: x["area"], reverse=True)
    unique_candidates = []

    for current in sorted_candidates:
        is_subset = False
        for existing in unique_candidates:
            if is_box_inside(current["coords"], existing["coords"]):
                is_subset = True
                # print(f"  > Removing subset: '{current['title']}' is inside '{existing['title']}'")
                break
        if not is_subset:
            unique_candidates.append(current)

    # 2. Group by Title
    candidates_by_title = {}
    for item in unique_candidates:
        title = item["title"]
        if title not in candidates_by_title:
            candidates_by_title[title] = []
        candidates_by_title[title].append(item)
    
    final_list = []
    ambiguous_items = []
    
    # --- MISSING LOGIC RESTORED HERE ---
    for title, items in candidates_by_title.items():
        if len(items) == 1:
            # If it appears only once, trust it automatically
            final_list.append(items[0])
        else:
            # If it appears multiple times, add to list for VLM checking
            ambiguous_items.extend(items)
            
    # --- CRITICAL CHECK ---
    if not ambiguous_items:
        print("  > No ambiguous titles found. Skipping VLM.")
        return final_list

    # Step 3: VLM Filter with Dynamic Colors
    print(f"  > Disambiguating {len(ambiguous_items)} items with VLM...")
    
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta"]
    id_map = {}
    
    for i, item in enumerate(ambiguous_items):
        box_id = i + 1
        id_map[box_id] = item
        c = item["coords"]
        
        color = colors[i % len(colors)]
        
        draw.rectangle([c["x1"], c["y1"], c["x2"], c["y2"]], outline=color, width=5)
        draw.rectangle([c["x1"], c["y1"]-30, c["x1"]+40, c["y1"]], fill=color)
        try:
            draw.text((c["x1"]+10, c["y1"]-25), str(box_id), fill="white") # Removed font_size for compatibility
        except:
            pass 

    temp_annotated_path = image_path.replace(".png", "_annotated.png")
    img.save(temp_annotated_path)
    
    b64_img = load_image_base64(temp_annotated_path)
    
    prompt = """
    You are a Blueprint Analyzer.
    I have highlighted text regions that appear multiple times on the sheet.
    
    ### TASK:
    Identify which IDs represent **ACTUAL SECTION TITLES**.
    
    ### VISUAL CLUES FOR A REAL TITLE:
    1. **Location:** Usually at the **BOTTOM** of a drawing/detail.
    2. **Style:** Usually **BOLD**, Uppercase, and Underlined.
    3. **Isolation:** Has empty white space around it.
    
    ### VISUAL CLUES FOR NOISE (REJECT THESE):
    1. **Inside a Drawing:** Text pointing to a specific part (e.g. a stud or beam).
    2. **Leader Lines:** Text with an arrow pointing to it.
    3. **Notes:** Text inside a paragraph.
    
    Return JSON: {"valid_ids": [1, 3]}
    """

    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
    ])
    
    try:
        response = llm_pro.invoke([msg])
        
        text_content = extract_text_from_response(response)
        
        # Use Regex to find JSON to be safe against extra text
        import re
        match = re.search(r'\{.*\}', text_content, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            valid_ids = data.get("valid_ids", [])
            print(f"  > VLM Selected IDs: {valid_ids}")
            
            for uid in valid_ids:
                if uid in id_map:
                    final_list.append(id_map[uid])
        else:
            print("  ! Could not find JSON in response. Keeping all.")
            final_list.extend(ambiguous_items)

        if os.path.exists(temp_annotated_path):
            os.remove(temp_annotated_path)

    except Exception as e:
        print(f"  ! Filtering failed: {e}. Keeping all ambiguous items.")
        final_list.extend(ambiguous_items)
        
    return final_list


def preprocess_image_inplace(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return False

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold
    processed = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    cv2.imwrite(image_path, processed)
    return True


def normalize_text(text):
    """Removes punctuation and extra spaces for better matching."""
    if not text: return ""
    return re.sub(r'[^a-zA-Z0-9]', '', text).upper()

def find_title(image_path: str):
    image_base64 = load_image_base64(image_path)
    prompt = """
    Look at this construction sheet.
    Extract ONLY the bold, uppercase TITLES found below the detail drawings.
    Examples: "TYP. WALL SECTION", "LADDER DETAIL", "SECTION A-A".
    Do NOT extract notes, dimensions, or material labels.
    Return just the titles, one per line.
    """
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
    ])
    response = llm_flash.invoke([message])
    titles = [line.strip() for line in response.content.split("\n") if line.strip()]
    return titles

def find_all_title_coordinates(words, titles):
    results = {} 
    
    # Pre-normalize PDF text for fuzzy matching
    for title in titles:
        norm_title = normalize_text(title)
        if not norm_title: continue
        
        title_words = title.split()
        n = len(title_words)
        
        candidates = []
        
        # Sliding window search
        for i in range(len(words) - n + 1):
            segment_text = "".join([w["text"] for w in words[i:i+n]])
            norm_segment = normalize_text(segment_text)
            
            if norm_title in norm_segment: 
                boxes = words[i:i+n]
                candidates.append({
                    "x1": min(w["x0"] for w in boxes),
                    "y1": min(w["top"] for w in boxes),
                    "x2": max(w["x1"] for w in boxes),
                    "y2": max(w["bottom"] for w in boxes)
                })
        
        if candidates:
            results[title] = candidates
            
    return results

def disambiguate_repeated_titles(image_path, title_coords_candidates):
    final_coords = {}
    for title, candidates in title_coords_candidates.items():
        if len(candidates) == 1:
            final_coords[title] = candidates[0]
        else:
            final_coords[title] = candidates[0] 
    return final_coords

def find_title_coordinates_from_image_and_pdf(pdf_path):
    results = {}
    os.makedirs("pages", exist_ok=True)
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):

            # 1. Render page image for LLM
            image_path = f"pages/page_{page_num}.png"
            page.to_image(resolution=300).save(image_path)

            # 2. LLM reads titles from image
            titles = find_title(image_path)   # List[str]

            # 3. Extract PDF words + coords
            words = page.extract_words(
                use_text_flow=True,
                keep_blank_chars=False
            )

            # 4. Collect all candidate coordinates for each title
            title_coords_candidates = find_all_title_coordinates(words, titles)

            print(f"The co-ordiantes we are getting is as : {title_coords_candidates}")
            
            results[f"page_{page_num}"] = title_coords_candidates

    return results


def extract_text_from_response(response):
    if isinstance(response.content, list):
        return "".join([part["text"] for part in response.content if "text" in part]).strip()
    return str(response.content).strip()

def get_sheet_number(image_path: str) -> str:
    image_b64 = load_image_base64(image_path)
    prompt = """
    Look at the BOTTOM RIGHT CORNER. Extract the SHEET NUMBER.
    Examples: "S-1.0", "S-3.2".
    Return ONLY the sheet number text. Do not write a sentence.
    """
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
    ])
    response = llm_pro.invoke([msg])
    return extract_text_from_response(response)


# Helper function for sementic_segementation

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

def is_box_inside(inner_box, outer_box):
    """Checks if inner_box is significantly overlapping or inside outer_box."""
    ix1, iy1, ix2, iy2 = inner_box["x1"], inner_box["y1"], inner_box["x2"], inner_box["y2"]
    ox1, oy1, ox2, oy2 = outer_box["x1"], outer_box["y1"], outer_box["x2"], outer_box["y2"]

    # Check intersection area
    x_left = max(ix1, ox1)
    y_top = max(iy1, oy1)
    x_right = min(ix2, ox2)
    y_bottom = min(iy2, oy2)

    if x_right < x_left or y_bottom < y_top:
        return False # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    inner_area = (ix2 - ix1) * (iy2 - iy1)
    
    # If >80% of the inner box is covered by the outer box, it's a duplicate/subset
    return (intersection_area / inner_area) > 0.8

def scale_coords_pdf_to_image(coords_dict, pdf_path, image_path):
    img = Image.open(image_path)
    img_width, img_height = img.size

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        pdf_width = page.width
        pdf_height = page.height

    scale_x = img_width / pdf_width
    scale_y = img_height / pdf_height

    scaled = {}
    
    # Iterate through titles
    for title, candidates_list in coords_dict.items():
        # candidates_list is a LIST of dicts: [{'x1':...}, {'x1':...}]
        
        scaled_candidates = []
        
        for c in candidates_list:
            scaled_c = {
                "x1": int(c["x1"] * scale_x),
                "y1": int(c["y1"] * scale_y),
                "x2": int(c["x2"] * scale_x),
                "y2": int(c["y2"] * scale_y),
            }
            scaled_candidates.append(scaled_c)
            
        # Store the list of scaled candidates back under the title
        scaled[title] = scaled_candidates

    return scaled