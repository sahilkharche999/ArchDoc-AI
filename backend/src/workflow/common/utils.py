import base64
import json
import os
import subprocess
import sys
from io import BytesIO

import fitz
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from google import genai
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pdf2image import convert_from_path

from src.logger import setup_logger
from src.workflow.common.schemas import DetailExtraction, DetailGroup, DetailMap

logger = setup_logger(__name__)

load_dotenv()
llm_pro = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")
llm_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


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
            simple_json = [{"id": i, "type": x["type"], "bbox": x.get("bbox"), "text_preview": x.get("text", "")[:50]} for
                        i, x in enumerate(json_data)]
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
    prompt = f"""
    You are a Layout Analysis Engine.
    I have parsed a PDF page into a list of items (Images, Text).
    
    ### INPUTS:
    1. **Layout Image:** Shows the visual arrangement.
    2. **JSON List:** The detected items with IDs.
    
    ### TASK:
    Group these items into **Logical Detail Units**.
    - A "Detail Unit" usually has a **Title** (Text) at the bottom.
    - Above the title, there are **Drawings** (Images) and **Notes** (Text).
    - **CRITICAL:** One Detail might have MULTIPLE images (e.g. a Plan View + a Section View + a Table). Group them all under the same Title.
    
    ### OUTPUT:
    Return a list of `DetailGroup` objects.
    - `detail_id`: Extract the number from the bubble (e.g. "7/S-3.2").
    - `image_files`: List the filenames of the images in this group (look at the JSON 'img_path').
    - `text_blocks`: List the full text of any notes in this group.
    """

    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{layout_image_b64}"}},
        {"type": "text", "text": f"JSON Items:\n{json_string}"}
    ])

    try:
        # Use Flash for layout mapping (it's fast and good at spatial grouping)
        result = llm_flash.with_structured_output(DetailMap).invoke([msg])
        logger.debug(f"[Layout] Mapping success | groups={len(result.groups)}")
        return result.groups
    except Exception as e:
        logger.error(f"[Layout] Mapping failed | error={str(e)}")
        return []


def extract_single_detail(group: DetailGroup, images_dir: str):
    """
    Analyzes a SINGLE detail group (specific images + text) to get the BOM.
    Used in the Floor plan agent 
    """
    logger.debug(f"   > Extracting BOM for {group.detail_id}...")

    payload = []

    # A. Prompt
    prompt = f"""
    You are a Senior Structural Detailer.
    Analyze this specific detail: **"{group.title}"** ({group.detail_id}).
    
    ### INPUTS:
    I have cropped the specific images and text for this detail.
    
    ### TASK:
    Extract the **Bill of Materials (BOM)** and **Fabrication Metrics**.
    
    1. **Read Leader Lines:** Look at the images. Extract material names EXACTLY as written.
    2. **Read Notes:** Look at the text blocks provided.
    3. **Define Logic:** Fixed vs Variable count.
    
    Return the `DetailExtraction` object.
    """
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


def get_sheet_number(image_path: str) -> str:
    """
    Get the sheet number present in the bottom right corner,used in storing the section detail information
    """
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

        lb_per_ft = material_data.get("lb_per_ft", item.get("lb_per_ft", 0))
        price = material_data.get("price_per_lb", 0)

        item["lb_per_ft"] = lb_per_ft

        item["total_weight_lbs"] = item["total_linear_feet"] * lb_per_ft * item.get("quantity", 1)

        item["charge_per_lb"] = price

        item["total_cost"] = item["total_weight_lbs"] * price

    return bom_items
