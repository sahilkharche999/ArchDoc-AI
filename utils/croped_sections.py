from langgraph.graph import StateGraph, START, END
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from typing import TypedDict, List
from dotenv import load_dotenv
import os
import fitz
import pdfplumber
from pdfplumber import open as pdf_open
from PIL import Image, ImageDraw, ImageFont
from utils.pdf_page_to_png import convert_specific_page_to_png
from utils.pdf_create import extracte_pdf
from pydantic import BaseModel
from typing import Literal
import cv2 
from PIL import Image
import pdfplumber
import cv2
import random
import json
load_dotenv()

llm=ChatGoogleGenerativeAI(
        model='gemini-2.5-pro'
)

class TitleChoice(BaseModel):
    choice: int  

def load_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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

def extract_text_from_gemini_response(response) -> str:
    """
    Safely extract text from Gemini / LangChain response.
    Handles string and multimodal list formats.
    """
    if isinstance(response.content, list):
        return " ".join(
            part.get("text", "")
            for part in response.content
            if isinstance(part, dict)
        ).strip()
    return str(response.content).strip()


def find_title(imgURL: str):
    image_base64 = load_image_base64(imgURL)
    prompt = """
You are looking at an architectural / structural drawing sheet.
Task:
- Extract ONLY the section titles present in the drawing Sheet.
Rules:
- Titles are bold, uppercase, and placed below each detail.
- Ignore dimensions, notes, callouts, and material labels.
- Do NOT explain anything.
- Do NOT guess if you cannot see.
- Return each title on new line as it is.
"""

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                },
            },
        ]
    )

    response = llm.invoke([message])


    if isinstance(response.content, list):
        text = "\n".join(
            part["text"]
            for part in response.content
            if isinstance(part, dict) and "text" in part
        )
    else:
        text = response.content

    titles = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    return titles



def extract_text_from_response(response):
    """Safely extracts text from a Gemini response object (String or List)."""
    if isinstance(response.content, list):
        return "".join([part["text"] for part in response.content if "text" in part]).strip()
    return str(response.content).strip()


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




def filter_candidate_coordinates(image_path, all_candidates_flat_list):
    """
    Filter candidates using :
    1. Geometric Subset Removal (Box inside Box)
    2. Height Thresholding (Removal small notes)
    3. VLM Visual Inspection (Final tie-breaker)
    """
    print("--- Filtering Noise Coordinates ---")

    for item in all_candidates_flat_list:
        c = item["coords"]
        item["area"] = (c["x2"] - c["x1"]) * (c["y2"] - c["y1"])
        item['height']=c["y2"]-c["y1"]

    sorted_candidates = sorted(all_candidates_flat_list, key=lambda x: x["area"], reverse=True)
    unique_candidates = []

    for current in sorted_candidates:
        is_subset = False
        for existing in unique_candidates:
            # Check if 'current' is inside 'existing'
            if is_box_inside(current["coords"], existing["coords"]):
                is_subset = True
                print(f"  > Removing subset: '{current['title']}' is inside '{existing['title']}'")
                break
        
        if not is_subset:
            unique_candidates.append(current)


    # Group by Title to identify singletons vs duplicates
    candidates_by_title = {}
    for item in unique_candidates:
        title = item["title"]
        if title not in candidates_by_title:
            candidates_by_title[title] = []
        candidates_by_title[title].append(item)
    
    final_list = []
    ambiguous_items = []
    
    # Step 1: Auto-Accept Singletons
    for title, items in candidates_by_title.items():
        if len(items) == 1:
            final_list.append(items[0])
            continue 

        max_height=max(item['height'] for item in items)

        big_items=[item for item in items if item['height']>=(max_height*0.8)]

        if len(big_items)<len(items):
            print(f"  > Removed {len(items) - len(big_items)} small text matches for '{title}'")

        if len(big_items)==1:
            final_list.append(big_items[0])
        else:
            ambiguous_items.extend(big_items)
         
    if not ambiguous_items:
        print("  > No ambiguous titles found. Skipping VLM.")
        return final_list

    # Step 2: VLM Filter with Dynamic Colors
    print(f"  > Disambiguating {len(ambiguous_items)} items with VLM...")
    
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # High-contrast colors for bounding boxes
    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta"]
    
    id_map = {}
    
    for i, item in enumerate(ambiguous_items):
        box_id = i + 1
        id_map[box_id] = item
        c = item["coords"]
        
        # Cycle through colors
        color = colors[i % len(colors)]
        
        # Draw Box
        draw.rectangle([c["x1"], c["y1"], c["x2"], c["y2"]], outline=color, width=5)
        
        # Draw ID Label (Background matches box color)
        draw.rectangle([c["x1"], c["y1"]-30, c["x1"]+40, c["y1"]], fill=color)
        try:
            draw.text((c["x1"]+10, c["y1"]-25), str(box_id), fill="white", font_size=20)
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
        # Use Flash for speed
        response = llm.invoke([msg])
        
        # Robust JSON extraction
        text_content = extract_text_from_response(response)
        json_str = text_content.replace("```json", "").replace("```", "").strip()
        data = json.loads(json_str)
        
        valid_ids = data.get("valid_ids", [])
        print(f"  > VLM Selected IDs: {valid_ids}")
        
        # Add validated items
        for uid in valid_ids:
            if uid in id_map:
                final_list.append(id_map[uid])
                
        # if os.path.exists(temp_annotated_path):
        #     os.remove(temp_annotated_path)

    except Exception as e:
        print(f"  ! Filtering failed: {e}. Keeping all ambiguous items.")
        final_list.extend(ambiguous_items)
        
    return final_list



def disambiguate_repeated_titles(image_path, title_coords_candidates):
    final_coords = {}

    for title, candidates in title_coords_candidates.items():
        if len(candidates) == 1:
            final_coords[title] = candidates[0]
            continue

        candidate_str = "\n".join(
            [f"{i+1}: x1={c['x1']}, y1={c['y1']}, x2={c['x2']}, y2={c['y2']}"
             for i, c in enumerate(candidates)]
        )

        prompt = f"""
You are looking at an architectural drawing sheet (image provided).

The title "{title}" appears multiple times.
Select the correct candidate.

Candidates:
{candidate_str}

Rules:
- Return ONLY a JSON object
- The number MUST be between 1 and {len(candidates)}

Example:
{{ "choice": 2 }}
"""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{load_image_base64(image_path)}"
                    },
                },
            ]
        )

        # Schema-enforced invoke
        response = llm.with_structured_output(TitleChoice).invoke([message])

        choice = response.choice - 1  # convert to 0-based

        # Absolute safety (should never trigger now)
        if not (0 <= choice < len(candidates)):
            choice = min(range(len(candidates)), key=lambda i: candidates[i]["y1"])

        final_coords[title] = candidates[choice]

    return final_coords


def extract_words_with_coords(pdf_path, page_num):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num - 1]
        return page.extract_words(
            use_text_flow=True,
            keep_blank_chars=False
        )

def find_all_title_coordinates(words, titles):
    results = {}  # title -> list of coordinates
    word_texts = [w["text"].upper() for w in words]

    for title in titles:
        title_words = title.upper().split()
        n = len(title_words)
        candidates = []

        for i in range(len(word_texts) - n + 1):
            if word_texts[i:i+n] == title_words:
                boxes = words[i:i+n]
                x0 = min(w["x0"] for w in boxes)
                x1 = max(w["x1"] for w in boxes)
                top = min(w["top"] for w in boxes)
                bottom = max(w["bottom"] for w in boxes)

                candidates.append({
                    "x1": x0,
                    "y1": top,
                    "x2": x1,
                    "y2": bottom
                })

        if candidates:
            results[title] = candidates

    return results

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

            # 5. Resolve duplicates using Gemini if needed
            # final_coords = disambiguate_repeated_titles(image_path, title_coords_candidates)

            results[f"page_{page_num}"] = title_coords_candidates

    return results


def draw_bounding_boxes_on_image(
    image_path: str,
    pdf_path: str,
    coords_dict: dict,
    outline_color: str = "red",
    line_width: int = 4
):
    """
    Scales PDF-space coordinates to image-space and draws bounding boxes
    on the image. Overwrites the same image.
    """

    # Load image
    img = Image.open(image_path).convert("RGB")
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)

    # Read PDF page size (single-page PDF)
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        pdf_width = page.width
        pdf_height = page.height

    # Scaling factors
    scale_x = img_width / pdf_width
    scale_y = img_height / pdf_height

    # Draw boxes
    for title, c in coords_dict.items():
        x1 = int(c["x1"] * scale_x)
        y1 = int(c["y1"] * scale_y)
        x2 = int(c["x2"] * scale_x)
        y2 = int(c["y2"] * scale_y)

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=outline_color,
            width=line_width
        )

    img.save(image_path)


def crop_sections_from_page(
    coords_dict: dict,          # PDF-space coords
    page_image_path: str,
    pdf_path: str,
    page_name: str,
    base_output_dir: str = "cropped_sections"
):
    """
    Crops architectural / structural sections from a page image
    using title coordinates extracted from PDF.
    """

    # ---------------------------
    # Create per-page output dir
    # ---------------------------
    page_output_dir = os.path.join(base_output_dir, page_name)
    os.makedirs(page_output_dir, exist_ok=True)

    # ---------------------------
    # Load image
    # ---------------------------
    page_image = Image.open(page_image_path)
    img_width, img_height = page_image.size

    # ---------------------------
    # Scale coordinates PDF → Image
    # ---------------------------
    scaled = scale_coords_pdf_to_image(
        coords_dict,
        pdf_path,
        page_image_path
    )

    # ---------------------------
    # Flatten the structure
    # ---------------------------
    flat_list = []
    for title, candidates in scaled.items():
        for c in candidates:
            flat_list.append({"title": title, "coords": c})

    # ==========================================
    # NEW STEP: FILTER NOISE BEFORE CROPPING
    # ==========================================
    clean_flat_list = filter_candidate_coordinates(page_image_path, flat_list)
    
    # If VLM filtered everything out by mistake, fallback to original
    if not clean_flat_list and flat_list:
        print("Warning: VLM filtered all items. Reverting to raw list.")
        clean_flat_list = flat_list

    # ---------------------------
    # Row-wise cropping (by y2) — IMAGE SPACE
    # ---------------------------
    cropped_sections = []

    # Get unique Y2 values from CLEAN list
    rows_y2 = sorted(set(item["coords"]["y2"] for item in clean_flat_list))
    prev_y = 0

    for row_idx, y2 in enumerate(rows_y2, start=1):

        if y2 <= prev_y:
            continue

        cropped_row = page_image.crop((0, prev_y, img_width, y2))

        # Find all items that belong to this row (matching Y2)
        items_in_row = [item for item in clean_flat_list if item["coords"]["y2"] == y2]
        
        # Sort them left-to-right by X1
        items_in_row_sorted = sorted(items_in_row, key=lambda x: x["coords"]["x1"])

        left_margin = 400
        right_margin = 200

        for col_idx, item in enumerate(items_in_row_sorted):
            title = item["title"]
            curr_x1 = item["coords"]["x1"]

            x_start = max(0, curr_x1 - left_margin)

            if col_idx < len(items_in_row_sorted) - 1:
                next_item = items_in_row_sorted[col_idx + 1]
                next_x1 = next_item["coords"]["x1"]
                x_end = max(x_start + 1, next_x1 - right_margin)
            else:
                x_end = img_width

            if x_end <= x_start:
                continue

            cropped_section = cropped_row.crop(
                (x_start, 0, x_end, cropped_row.height)
            )

            safe_title = title.replace("/", "_").replace(" ", "_")
            # Add index to filename to handle duplicates (e.g. "SECTION_A_1.png")
            save_path = os.path.join(
                page_output_dir,
                f"{safe_title}_{col_idx}.png"
            )

            cropped_section.save(save_path)

            cropped_sections.append({
                "title": title,
                "image_path": save_path,
                "coords": (x_start, prev_y, x_end, y2)
            })

        prev_y = y2 + 110  # spacing buffer

    return cropped_sections
    
# --- Run the pipeline ---

if __name__=="__main__":
        pdf_path ="utils/all_section.pdf"
        output_path='section'
        try :
          if not os.path.exists(output_path):
              os.mkdir(output_path)
          doc=fitz.open(pdf_path)
          for i,page in enumerate(doc):
              page_image_path=f'section/page_{i}.png'
              page_section_pdf=f'section/page_{i}.pdf'
              extracte_pdf(pdf_path,page_section_pdf,i+1,i+1)
              convert_specific_page_to_png(page_section_pdf,0,page_image_path)
              coords_dict = find_title_coordinates_from_image_and_pdf(page_section_pdf)
              
              page_key = list(coords_dict.keys())[0]
              scale_coords_dict=scale_coords_pdf_to_image(coords_dict[page_key],page_section_pdf,page_image_path)
              print(scale_coords_dict)
              print("-------")
            #   draw_bounding_boxes_on_image(page_image_path,page_section_pdf,coords_dict[page_key])
              crop_sections_from_page(coords_dict[page_key],page_image_path,page_section_pdf,f"page_{i}",)
        except Exception as e:
         raise ValueError(e)


