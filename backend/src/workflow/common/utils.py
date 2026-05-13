import base64
import json
import os
import subprocess
import sys
import re
from io import BytesIO
import pdfplumber
import fitz
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from google import genai
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pdf2image import convert_from_path
from src.infrastructure.graph_db import graph_db
from src.infrastructure.symbol_detection import detect_and_read_symbols 
from src.workflow.workflows.estimation.prompt import (
    prompt_for_node_process_plans,prompt_for_extract_single_detail,prompt_for_map_page_layout,prompt_for_classify_image_as_plan_detail
    )
from src.workflow.common.schemas import (
    IngestionOutput ,FinalEstimation
    )
from src.logger import setup_logger
from src.workflow.common.schemas import DetailExtraction, DetailGroup, DetailMap

logger = setup_logger(__name__)

load_dotenv()
llm_pro = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")
llm_25_pro = ChatGoogleGenerativeAI(model="gemini-2.5-pro") 
llm_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def _bbox_overlap_ratio(b1, b2):
    """Returns what fraction of b1's area is covered by b2."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
    return inter / b1_area if b1_area > 0 else 0.0

def map_page_layout(pdf_layout_path: str, json_path: str, images_dir: str):
    """
    Uses VLM to look at the full page layout and group items into 'Detail Units'.
    Returns a list of DetailGroup objects.
    Used in Process Plan agent
    """
    logger.debug(f"[Layout] Mapping started | pdf={pdf_layout_path}")
    try:
    # 1. Load Context
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            # Simplify JSON for prompt (just types and bboxes)
        if isinstance(json_data, list) and len(json_data) > 0 and isinstance(json_data[0], list):
            json_data = json_data[0]
        simple_json = [{"id": i, "type": x["type"], "bbox": x.get("bbox"), "text_preview": x.get("text", "")[:50],"img_path": x.get("img_path", None)} 
                          for i, x in enumerate(json_data)]
        json_string = json.dumps(simple_json, indent=2)
    except Exception as e:
        logger.error(f"[Layout] Failed to load JSON | path={json_path} | error={str(e)}")
        return []


    # 2. Convert Layout PDF to Image
    try:
        layout_images = convert_from_path(pdf_layout_path)
        layout_image_b64 = image_to_base64(layout_images[0])
    except:
        logger.error(f"[Layout] PDF to image conversion failed | path={pdf_layout_path} | error={str(e)}")
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
        logger.debug(f"[Layout] Mapping success | groups={result.groups}")
        return result.groups
    except Exception as e:
        logger.error(f"[Layout] Mapping failed | error={str(e)}")
        return []
    

def classify_image_as_plan(image_path):

    prompt =prompt_for_classify_image_as_plan_detail()

    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{load_image_base64(image_path)}"}}
    ])

    try:
        result = llm_flash.invoke([msg])
        text = result.content.strip()

        # Parse JSON response
        try:
            # Strip markdown fences if present
            clean = text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
            img_type = parsed.get("type", "INDEPENDENT_DETAIL").upper()
        except Exception:
            # Fallback: keyword search in raw text
            text_upper = text.upper()
            if "PLAN_VIEW" in text_upper:
                img_type = "PLAN_VIEW"
            elif "DEPENDENT_DETAIL" in text_upper:
                img_type = "DEPENDENT_DETAIL"
            elif "IGNORE" in text_upper:
                img_type = "IGNORE"
            else:
                img_type = "INDEPENDENT_DETAIL"

        logger.debug(f"   > Classification result: {img_type} | image={os.path.basename(image_path)}")
        return img_type

    except Exception as e:
        logger.error(f"   ! classify_image_as_plan failed: {e}")
        return "INDEPENDENT_DETAIL" 

def extract_single_detail(group: DetailGroup, images_dir: str, temp_plan_like_details: list,temp_dependent_detail_images:list, sheet_number: str, page_num: int):
    """
    Analyzes a SINGLE detail group (specific images + text) to get the BOM.
    Used in the Floor plan agent 
    """
    logger.debug(f"   > Extracting BOM for {group.detail_id}...")

    payload = []
    plan_images = []
    detail_images = []
    dependent_detail_images=[]

      # STEP 1 — Classify each image
    for img_file in group.image_files:
        fname = os.path.basename(img_file)
        full_path = os.path.join(images_dir, fname)

        if not os.path.exists(full_path):
            continue

        img_type = classify_image_as_plan(full_path)

        if img_type == "PLAN_VIEW":
            plan_images.append(full_path)
        elif img_type == "DEPENDENT_DETAIL":
            dependent_detail_images.append(full_path)
        else:
            detail_images.append(full_path)

     # STEP 2 — If ANY plan image → store & exit
    if len(plan_images) > 0:
        logger.debug(f"Plan-like detected in group {group.detail_id}")

        temp_plan_like_details.append({
            "detail_id": group.detail_id,
            "sheet": sheet_number,
            "image_path": plan_images, 
            "page": page_num,
            "title": group.title 
        })
    
    if len(dependent_detail_images)>0:
        logger.debug(f"dependent_detail_images detected in group {group.detail_id}")
        temp_dependent_detail_images.append({
            "detail_id": group.detail_id,
            "sheet": sheet_number,
            "image_path": dependent_detail_images, 
            "page": page_num,
            "title": group.title 
        }
        )
    if len(detail_images) == 0:
        return None 

    # A. Prompt
    prompt = prompt_for_extract_single_detail(group_title=group.title,group_detail_id=group.detail_id)
    payload.append({"type": "text", "text": prompt})

    # B. Add Specific Images
    for img_path in detail_images:
        b64 = load_image_base64(img_path)
        payload.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })

    # C. Add Specific Text
    if group.text_blocks:
        clean_text = "\n".join([t.strip() for t in group.text_blocks if t.strip()])
        payload.append({"type": "text", "text": f"NOTES:\n{clean_text}"})

    try:
        # Use Pro for reading the engineering text
        result = llm_pro.with_structured_output(DetailExtraction).invoke([HumanMessage(content=payload)])
        return result
    except Exception as e:
        logger.error(f"   ! Extraction failed for {group.detail_id}: {e}")
        return None


def crop_union_tables(json_path, image_path, output_dir="debug_crops"):
    """
    Used in the agnet 2 floor plan , where it task is to combine the text+image co-ordiante and crop combine table,
    which make sure healing and content comes in crop
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.debug(f"[Crop] Start union cropping | image={image_path}")

    if not os.path.exists(image_path):
        logger.error(f"Error: Image not found at {image_path}")
        return

    full_img = Image.open(image_path)
    img_w, img_h = full_img.size
    logger.debug(f"Loaded Image: {img_w}x{img_h}")

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

    logger.debug(f"Detected Scale Factor: X={scale_x:.2f}, Y={scale_y:.2f}")

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
            except Exception as e:
                logger.warning(f"[Crop] Failed to extract title text | index={i} | error={str(e)}")
                safe_title = f"Title_{i}"

            logger.debug(f"Checking Title: '{safe_title}'...")

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
                logger.debug(f"  -> MATCH! Found '{body_item['type']}' below (Gap: {min_gap:.1f})")

                # Calculate Union Box
                union_x1 = min(bbox[0], body_bbox[0]) - 60  # Padding for Symbol
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
                    logger.debug(f"  -> Saved Union Crop: {save_path}")

                    # Mark both as processed
                    processed_indices.add(i)
                    processed_indices.add(best_match_idx)

                except Exception as e:
                    logger.error(f"[Crop] Crop failed | title={safe_title} | error={str(e)}")

        # 4. If it's a Table/List that wasn't merged (Orphan)
        elif item_type in ["table", "list"] and i not in processed_indices:
            # Just use the existing image if available, or crop it fresh
            # This handles tables that MinerU found perfectly without a separate title
            pass


def convert_specific_page_to_png(pdf_path, page_num, output_image_path, dpi=300):
    try:
        # open the pdf into one container
        doc = fitz.open(pdf_path)
        if 0 <= page_num < doc.page_count:
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            pix.save(output_image_path)
            logger.debug(f"Successfully converted page {page_num} to {output_image_path}")
        else:
            logger.error(f"Error: Page number {page_num} is out of range.")
        doc.close()
    except Exception as e:
        logger.error(f"An error occurred: {e}")


def minerU_pdf_creating_extration(pdf_path: str, output_dir: str, backend_type: str):
    os.makedirs(output_dir, exist_ok=True)
    mineru_bin = os.path.join(os.path.dirname(sys.executable), "mineru")
    cmd = [
        mineru_bin,   
        "-p", pdf_path,
        "-o", output_dir,
        "-m", "auto",
        "-l", "en",
        "-t", "true",
        "-f", "false",
        "-b", backend_type
    ]

    subprocess.run(cmd, check=True)


def get_valid_materials_list(excel_path):
    """
    Load the Excel sheet and return the last 'Option' tab which content the material size + linear feet + cost
    """
    try:
        df = pd.read_excel(excel_path, sheet_name="Options")
        return df.iloc[:, 0].dropna().astype(str).tolist()
    except Exception as e:
        logger.error(f"Failed to read Excel file: {excel_path} | error={str(e)}")
        raise


def load_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_to_base64(image_obj):
    buff = BytesIO()
    image_obj.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def extract_text_from_response(response):
    """Used in the get sheet number function and task is to retunr json strucutred"""
    if isinstance(response.content, list):
        return "".join([part["text"] for part in response.content if "text" in part]).strip()
    return str(response.content).strip()

def extract_sheet_candidate(text: str) -> str:
    if not text:
        return ""

    text = text.upper()

    patterns = [
        r"[A-Z]+-[A-Z]+-\d+\.?\d*",   # ST-DT-0029
        r"[A-Z]-\d+\.?\d*"            # S-2.0
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)

    return text  # fallback

def normalize_sheet(sheet: str,prefix: str = "") -> str:
    if not sheet:
        return ""
    sheet=sheet.strip().upper()
    prefix = prefix.strip().upper()
    if prefix and sheet.startswith(prefix):
        sheet = sheet[len(prefix):].lstrip("-_")

    sheet = re.sub(r"\s*-\s*", "-", sheet)
    sheet = re.sub(r"[^A-Z0-9\-.]", "", sheet)

    parts = sheet.split("-")
    if len(parts) >= 3:
        return "-".join(parts[-3:])
    return sheet

def normalize_detail_key(detail_id: str, sheet_number: str) -> str:
    """
    If detail_id has a slash, the part after slash is the sheet reference.
    If that sheet reference doesn't match the current sheet_number,
    replace it with the current sheet_number.
    But only replace if the referenced sheet is clearly wrong
    (i.e. it's a different sheet family entirely).
    """
    if "/" not in detail_id:
        return f"{detail_id}/{sheet_number}"
    
    label, ref_sheet = detail_id.rsplit("/", 1)
    
    # If the sheet reference in the detail_id matches current sheet → keep as is
    if ref_sheet.upper() == sheet_number.upper():
        return detail_id
    
    # If it doesn't match, the VLM read the wrong sheet from the image
    # Use current sheet_number instead
    return f"{label}/{sheet_number}"

def get_sheet_number(image_path: str,sheet_prefix: str = "") -> str:
    """
    Get the sheet number present in the bottom right corner,used in storing the section detail information
    """
    image_b64 = load_image_base64(image_path)
    prompt = """
    Extract the SHEET NUMBER from the drawing.

    Instructions:
    - Look primarily in the title block (usually bottom right).
    - The sheet number can be in formats like:
        S-2.0
        A-1.1
        ST-DT-0029
        FA31137-ST-DT-0029
    - Return ONLY the sheet number text exactly as written.
    - Do NOT add any explanation.
    - Do NOT include labels like "Sheet No", "Drawing No", etc.
    - Do NOT return extra words.
    """
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
    ])
    response = llm_pro.invoke([msg])
    raw = extract_text_from_response(response)
    clean = extract_sheet_candidate(raw)
    normalized = normalize_sheet(clean,prefix=sheet_prefix)
    return {
        "full": raw,
        "normalized": normalized
    }



def normalize_material(name: str):
    return name.replace(" ", "").upper()


def load_material_weights(excel_path):
    df = pd.read_excel(excel_path, sheet_name="Options")

    material_lookup = {}

    for _, row in df.iterrows():
        material = normalize_material(str(row["Material Size Description"]))
        weight = float(row["Lbs/ft"])
        price = float(row["$ Charge per lb"])

        material_lookup[material] = {
            "lb_per_ft": weight,
            "price_per_lb": price
        }

    return material_lookup


def enrich_bom_with_pricing(bom_items, material_lookup):
    for item in bom_items:
        material = normalize_material(item["material_size"])

        material_data = material_lookup.get(material, {})

        if not material_data:
            print(f"[WARNING] Material not found in lookup: {material}")

        lb_per_ft = material_data.get("lb_per_ft", 0) if material_data else 0
        price = material_data.get("price_per_lb", 0) if material_data else 0

        item["lb_per_ft"] = lb_per_ft

        item["total_weight_lbs"] = item["total_linear_feet"] * lb_per_ft * item.get("quantity", 1)

        item["charge_per_lb"] = price

        item["total_cost"] = item["total_weight_lbs"] * price

    return bom_items

def normalize_pdf_orientation(input_pdf, output_pdf, page_angles):
    doc = fitz.open(input_pdf)

    for i, page in enumerate(doc):
        angle = page_angles.get(i, 0)
        current_rotation = page.rotation
        new_rotation = (current_rotation + angle) % 360
        page.set_rotation(new_rotation)

    doc.save(output_pdf)
    doc.close()

    return output_pdf

def classify_group_image(image_path):
    prompt = prompt_for_node_process_plans()

    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{load_image_base64(image_path)}"}}
    ])

    result = llm_flash.with_structured_output(IngestionOutput).invoke([msg])

    return result.type
    
def is_detail_ref(text: str) -> bool:
    if not text:
        return False

    text = text.strip().upper()

    # Matches: 1/S-3.2, 12/S-10, A/B.
    pattern = r"^[\dA-Za-z]+/.+$"

    return bool(re.match(pattern, text))

def _enrich_symbols(raw_symbols, project_id, sheet_number):
    """Shared helper: semantic search + schedule resolution for a symbol list."""
    enriched = []
    for sym in raw_symbols:
        query_text = sym.get("text_content", "").strip().upper()
        definition = None

        if is_detail_ref(query_text):
            definition = graph_db.get_definition_by_id(query_text, project_id)

        elif re.match(r"^(cir|hex)-(\d+)$", query_text, re.IGNORECASE):
            m = re.match(r"^(cir|hex)-(\d+)$", query_text, re.IGNORECASE)
            bare_number = m.group(2)
            prefixed = query_text.upper()
            logger.info(f"Using SCHEDULE LOOKUP for {query_text} → trying {prefixed} then {bare_number} on sheet {sheet_number}")
            # Try prefixed first (keyed notes stored as "HEX-24")
            definition = graph_db.get_definition_by_id(prefixed, project_id, sheet_number=sheet_number)
            if not definition:
                # Fallback to bare number (kettle cover stored as "24")
                definition = graph_db.get_definition_by_id(bare_number, project_id, sheet_number=sheet_number)

        else:
            matches = graph_db.semantic_search(query_text, project_id, sheet_number=None, limit=1)
            if matches:
                definition = matches[0]

        if definition and definition.get("BOM"):
            logger.info(f"DEFINITION FOUND: {definition}")
            for item in definition["BOM"]:
                logger.debug(f"Item : {item}")
                rule_text = item.get("qty_rule", "")
                mat_text  = item.get("item_name", "") or ""
                text_blob = f"{rule_text} {mat_text}".lower()
                schedule_keywords = ["schedule", "see plan", "see sched", "per schedule"]

                if any(k in text_blob for k in schedule_keywords) and mat_text:
                    logger.debug(f"    > Resolving Reference: {mat_text}")
                    sub_matches = graph_db.semantic_search(mat_text, project_id, sheet_number=None, limit=1)
                    if sub_matches:
                        s = sub_matches[0]
                        item["linked_schedule_data"] = {
                            "schedule_id":   s.get("ID"),
                            "schedule_name": s.get("Name"),
                            "columns":       s.get("Columns"),
                            "rows":          s.get("Rows"),
                            "sheet":         s.get("Sheet"),
                        }
                        logger.debug(f"      -> Found schedule: {s.get('ID')}")

        sym["linked_definition"] = definition
        enriched.append(sym)
        logger.debug(f"    > Enriched {len(enriched)} symbols with Graph Data.")
    return enriched


def _run_symbol_estimation(img_path, enriched_symbols,group_title="", group_detail_id=""):
    """Shared helper: call LLM with enriched symbols and return BOM items."""
    dependency_context = """
             ### ADDITIONAL CONTEXT — RESOLVED DEPENDENCIES

                    You are given pre-resolved data from referenced callouts.

                    Use this ONLY if:
                    - The image contains matching callout references (e.g., G9, K1)
                    - The dependency data corresponds to that reference

                    If no matching callout is visible:
                    → IGNORE this section completely

                    DO NOT double count materials.
                    DO NOT assume relationships unless clearly referenced.
    """
    prompt = prompt_for_extract_single_detail(group_title,group_detail_id)
    b64    = load_image_base64(img_path)
    msg    = HumanMessage(content=[
        {"type": "text", "text": dependency_context},
        {"type": "text", "text": prompt},
        {"type": "text", "text": f"DEPENDENCY DATA:\n{json.dumps(enriched_symbols)}"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ])
    try:
        result = llm_pro.with_structured_output(DetailExtraction).invoke([msg])
        return result
    except Exception as e:
        logger.error(f"Symbol estimation failed: {e}")
        return None

def _bom_item_to_material_dict(item):
    """Convert BillOfMaterialItem to MaterialItem-compatible dict for graph storage."""
    d = item.model_dump() if hasattr(item, 'model_dump') else dict(item)
    # BillOfMaterialItem uses material_size, MaterialItem uses item_name
    if "item_name" not in d:
        d["item_name"] = d.get("material_size", "")
    if "qty_rule" not in d:
        d["qty_rule"] = f"FIXED: {d.get('quantity', 1)}"
    return d

def _extract_text_references_as_symbols(img_path: str, sheet_number: str) -> list:

    prompt = """You are reading a structural engineering detail drawing.
            Your job is to find ALL detail references in this image — both graphical and text-based.

            TYPE 1 — GRAPHICAL CALLOUT BUBBLES:
            A circle divided horizontally, number on top, sheet reference on bottom.
            Examples: circle showing "3" over "S3-01" → extract: 3/S3-01
                        circle showing "G9" over "S522" → extract: G9/S522

            TYPE 2 — TEXT-BASED REFERENCES in leader lines:
            Leader line text containing PER or SEE followed by a detail number and sheet.
            "HAIRPIN PER 301/ST10"         → extract: 301/ST10
            "BEARING PAD PER 303/ST10"     → extract: 303/ST10
            "SEE DETAIL 5/S522"            → extract: 5/S522
            "PER DETAIL 2/S3-01"           → extract: 2/S3-01
            "PER 301"                      → extract: 301   (no sheet visible)
            "SEE DETAIL B"                 → extract: B     (no sheet visible)

            WHAT TO IGNORE — do NOT return these:
            - Material callouts like "L3X3X1/4", "W8X13", "PL 1/2"
            - Weld size annotations like "3/16", "1/4 FILLET"
            - Dimension text like "8'-0\"", "1'-6\""
            - "SEE PLAN" or "SEE SCHEDULE" with no detail number
            - The detail's OWN title bubble at the bottom (e.g. "304" over "ST10" as the title)

            Rules:
            - Output ONLY the extracted reference IDs, one per line
            - If NUMBER/SHEET is present → output exactly: NUMBER/SHEET
            - If only NUMBER or LETTER with no sheet → output just: NUMBER or LETTER
            - No explanation, no extra text, no bullet points
            - If nothing found → output: NONE
            """

    try:
        b64 = load_image_base64(img_path)
        msg = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        ])
        response = llm_flash.invoke([msg])
        raw = response.content.strip() if isinstance(response.content, str) else response.content[0]["text"].strip()
        logger.debug(f"Text-reference scan raw response | image={os.path.basename(img_path)} | raw={raw!r}")

        if raw.upper() == "NONE" or not raw:
            logger.info(f"Text-reference scan: no references found | image={os.path.basename(img_path)}")
            return []

        synthetic_symbols = []
        for line in raw.splitlines():
            ref = line.strip().upper()
            if not ref or ref == "NONE":
                continue

            # Case 1 — already has NUMBER/SHEET → use directly
            if re.match(r'^[\dA-Za-z]+/[\w\-\.]+$', ref):
                text_content = ref

            # Case 2 — only NUMBER or LETTER, no sheet → append current sheet_number
            elif re.match(r'^[\dA-Za-z]+$', ref) and len(ref) >= 1:
                text_content = f"{ref}/{sheet_number}"
                logger.debug(f"No sheet in reference '{ref}' — appended sheet: {text_content}")

            else:
                logger.debug(f"Skipping unrecognised reference line: {ref!r}")
                continue

            synthetic_symbols.append({
                "shape": "text_reference",
                "text_content": text_content,
                "bbox": [0, 0, 0, 0]
            })
            logger.debug(f"Synthetic symbol created | text_content={text_content}")

        logger.info(
            f"Text-reference scan completed | image={os.path.basename(img_path)} "
            f"| found={len(synthetic_symbols)}"
        )
        return synthetic_symbols

    except Exception as e:
        logger.error(f"Text reference scan failed | image={img_path} | error={str(e)}")
        return []
    
def _process_dependent_details(items, detail_library,sheet_number, config, state):
    project_id = config["configurable"]["thread_id"]
    for plan in items:
        img_paths  = plan["image_path"] if isinstance(plan["image_path"], list) else [plan["image_path"]]
        plan_sheet = plan["sheet"]

        for img_path in img_paths:
            logger.debug(f"Processing dependent detail: {img_path}")
            

            raw_symbols = _extract_text_references_as_symbols(img_path, plan_sheet)
            if not raw_symbols:
                logger.info(
                    f"No references found (graphical or text) — skipping | "
                    f"image={os.path.basename(img_path)}"
                )
                continue

            logger.info(
                f"References found | image={os.path.basename(img_path)} "
                f"| count={len(raw_symbols)}"
            )

            enriched = _enrich_symbols(raw_symbols, project_id, plan_sheet)
            detail_result = _run_symbol_estimation(
                img_path,
                enriched,
                group_title=plan.get("title", ""),
                group_detail_id=plan.get("detail_id", "")
            )
            

            if detail_result and detail_result.materials:
                # normalized = [_bom_item_to_material_dict(m) for m in bom_items] 
                normalized=[m.model_dump() for m in detail_result.materials]
                inherited_materials = []
                for sym in enriched:
                    defn = sym.get("linked_definition")
                    if not defn or not defn.get("BOM"):
                        continue
                    sub_detail_id = defn.get("ID", "unknown")
                    for sub_mat in defn["BOM"]:
                        # Tag inherited materials so they're traceable
                        inherited_item = dict(sub_mat)
                        inherited_item["notes"] = (
                            f"{inherited_item.get('notes', '')} | "
                            f"INHERITED_FROM: {sub_detail_id}"
                        ).strip(" |")
                        # Only add if not already present by item_name
                        already_present = any(
                            m.get("item_name") == inherited_item.get("item_name")
                            and m.get("notes", "") == inherited_item.get("notes", "")
                            for m in normalized
                        )
                        if not already_present:
                            inherited_materials.append(inherited_item)
                            logger.debug(
                                f"Inherited material {inherited_item.get('item_name')} "
                                f"from {sub_detail_id} into {plan['detail_id']}"
                            )
                all_materials = normalized + inherited_materials
                merged_fabrication = detail_result.fabrication.model_dump()
                for sym in enriched:
                    defn = sym.get("linked_definition")
                    if defn and defn.get("fabrication"):
                        sub_fab = defn["fabrication"]
                        merged_fabrication["bolt_count"] += sub_fab.get("bolt_count", 0)
                        merged_fabrication["hole_count"] += sub_fab.get("hole_count", 0)
                        merged_fabrication["weld_inches"] += sub_fab.get("weld_inches", 0.0)
                        
                graph_db.add_detail_bom(
                    project_id=project_id,
                    detail_key=plan['detail_id'] if "/" in plan['detail_id'] else f"{plan['detail_id']}/{plan_sheet}",
                    title=plan.get("title", "PLAN_RESOLUTION"),
                    materials_list=all_materials, 
                    fabrication=merged_fabrication,
                    page_num=plan["page"],
                    sheet_number=plan_sheet
                )


def _process_plan_like_details(items, detail_library, sheet_number,config, state):
    """Same as _process_dependent_details — separated for clarity."""
    _process_dependent_details(items, detail_library,sheet_number, config, state)


def normalize_detail_id(detail_id: str) -> str:
    """Fix common OCR errors in detail IDs from map_page_layout."""
    # Fix: 2/55-02 → 2/S5-02, 3/55-01 → 3/S5-01 etc.
    # Pattern: after slash, if starts with digit(s) followed by hyphen, 
    # it's likely S was misread as 5
    detail_id = re.sub(r'(?<=/)(5)(\d+-\d+)', r'S\2', detail_id)
    return detail_id
