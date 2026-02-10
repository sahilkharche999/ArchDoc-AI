import base64
import cv2
import os
import re 
from typing import  List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import pdfplumber
import json
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

load_dotenv()

# --- 1. SETUP MODELS ---
llm_pro = ChatGoogleGenerativeAI(model="gemini-3-pro-preview") 
llm_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite") 


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
    prompt = "Look at the BOTTOM RIGHT CORNER. Extract the SHEET NUMBER (e.g., S-3.2)."
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