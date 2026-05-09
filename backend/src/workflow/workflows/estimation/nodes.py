import os
import cv2
import json
import fitz
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt
from src.workflow.common.schemas import DetailExtraction
import pdfplumber
from pypdf import PdfReader, PdfWriter
from src.workflow.common.state import ProjectState
from src.workflow.common.schemas import (
    DrawingTypeResponse,
    FinalEstimation,
    TextRulesExtraction,
    IngestionOutput 
    )
from src.workflow.workflows.estimation.prompt import (
    prompt_for_node_classify_pages,
    prompt_for_node_process_plans,
    prompt_node_process_text_rules,
    prompt_for_agent_4_merger,
    prompt_for_extract_single_detail
    )
from src.workflow.common.utils import (
    crop_union_tables,
    map_page_layout,
    extract_single_detail,
    classify_image_as_plan,
    get_valid_materials_list,
    load_image_base64,
    get_sheet_number,
    convert_specific_page_to_png,
    load_material_weights,              
    minerU_pdf_creating_extration,       
    is_detail_ref ,
    _process_plan_like_details,
    _process_dependent_details,
    enrich_bom_with_pricing,
    normalize_detail_id,
    normalize_detail_key,
    _bbox_overlap_ratio
    )

from src.infrastructure.graph_db import graph_db
from src.infrastructure.symbol_detection import detect_and_read_symbols 
from src.logger import setup_logger
from src.db.update_jobs_status import update_job_status,update_job_progress

logger = setup_logger(__name__)
load_dotenv()
# --- 1. SETUP MODELS ---
llm_pro = ChatGoogleGenerativeAI(model="gemini-3-pro-preview") 
llm_25_pro = ChatGoogleGenerativeAI(model="gemini-2.5-pro") 
llm_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite") 



# ---  AGENT 0: PAGE CLASSIFY ---
def node_classify_pages(state: ProjectState):
    """
    Classifies pages in a PDF document by converting them to images and analyzing their content.
    
    Args:
        state (ProjectState): The current project state containing pdf_path and output_dir
        
    Returns:
        dict: Updated state with page classification results
        - "text": If the page contains mostly Notes, Schedules, Tables, or Specifications.
        - "floor": If the page shows a Plan View, Foundation Plan, or Roof Framing Plan.
        - "section": If the page shows Detail Drawings, Wall Sections, or Connection Cuts.
    """
    logger.info("--- NODE 0 : Classifying Pages ---")
    pdf_path = state["pdf_path"]
    output_dir=state['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    cache_path = f"{output_dir}/page_map_cache.json"
    if os.path.exists(cache_path):
        # Replay path — load from cache, skip all LLM calls
        with open(cache_path) as f:
            page_map = {int(k): v for k, v in json.load(f).items()}
        logger.info("Page map loaded from cache, skipping re-classification")
    else:
        page_map = {}
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
        for page_num in range(total_pages):
            temp_img_path = f"{state['output_dir']}/temp_page_{page_num}.png"
            convert_specific_page_to_png(pdf_path, page_num, temp_img_path, dpi=300)
            prompt=prompt_for_node_classify_pages()

            image_b64 = load_image_base64(temp_img_path)
            msg = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ])
            try:
                result = llm_flash.with_structured_output(DrawingTypeResponse).invoke([msg])
            except Exception as e:
                logger.error(f"Failed to classify page {page_num} | error={str(e)}")
                raise
            
            valid_types = {"text", "floor", "section"}
            drawing_type = result.drawing_type.strip().lower()
            if drawing_type not in valid_types:
                logger.warning(f"Unexpected drawing_type={drawing_type} on page {page_num}")
                continue
            page_map[page_num] = result.drawing_type
            os.remove(temp_img_path)
            logger.debug(f"Page Index {page_num}: {result.drawing_type}")
        logger.info(f"Page classification completed | total_pages={total_pages}")

        with open(cache_path, "w") as f:
            json.dump(page_map, f)
            
    review_data = {
        "type": "classify_review",
        "page_map": page_map
    }
    resume_value=interrupt(review_data)
    if resume_value is None:
            return state
    corrected_page_map = resume_value.get("corrected_page_map", page_map)
    corrected_page_map = {int(k): v for k, v in corrected_page_map.items()}
    return {"page_map": corrected_page_map}


# ---  AGENT 1: TEXT PROCESSOR ---
def node_process_text_rules(state: ProjectState,config):
    """
    Extracts text content and schedule rules from text pages using minerU.
    
    Workflow:
    1. Extract individual PDF pages from the main document
    2. Apply minerU to extract structured content including column-wise data
    3. Parse extracted markdown using LLM to identify schedule rules and symbols
    4. Store extracted rules in Neo4j graph database with embeddings for semantic search
    
    Args:
        state (ProjectState): Project state containing:
            - pdf_path: Path to the source PDF
            - output_dir: Directory for temporary outputs
            - page_map: Dictionary mapping page numbers to page types (text, floor, section)
            - general_rules: Accumulated notes from all processed pages
    
    Returns:
        dict: Updated state with:
            - general_rules: Concatenated notes from all text pages processed
    """
    logger.info("--- NODE 1: Processing Text Rules ---")
    text_pages = [p for p, t in state["page_map"].items() if t == "text"]
    
    for page_num in text_pages:
        # 1. Extract Single Page PDF
        logger.debug(f"[Page {page_num}] Start processing")
        page_pdf_path = f"{state['output_dir']}/notes_{page_num}.pdf"
        try:
            reader = PdfReader(state["pdf_path"])
            writer = PdfWriter()
            writer.add_page(reader.pages[page_num])
            with open(page_pdf_path, "wb") as f: writer.write(f)
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise

        # 2. Run MinerU (Assuming this function works and saves to the path below)
        logger.debug(f"   > Running MinerU on Page {page_num}...")
        minerU_pdf_creating_extration(page_pdf_path, state["output_dir"],"pipeline")

        # 3. Read the Markdown File
        # Note: Adjust path logic if MinerU creates subfolders differently
        md_file_path = f"{state['output_dir']}/notes_{page_num}/auto/notes_{page_num}.md"
        
        try:
            with open(md_file_path, "r", encoding="utf-8") as f:
                markdown_content = f.read()
        except FileNotFoundError:
            logger.error(f"   ! Markdown file not found: {md_file_path}")
            continue

        # 4. The Advanced Prompt
        logger.debug("   > Calling LLM to parse Rules...")
        
        prompt=prompt_node_process_text_rules(markdown_content)

        msg = HumanMessage(content=prompt)
        
        try:
            result = llm_flash.with_structured_output(TextRulesExtraction).invoke([msg])

            # Store Rules in Graph
            for section in result.sections:
                section_name = section.section_name
                for rule in section.rules:
                    graph_db.add_text_rule(
                        project_id=config["configurable"]["thread_id"],
                        section_name=section_name,
                        rule_number=rule.rule_number,
                        text=rule.text,
                        page_num=page_num
                    )
                    logger.debug( f"   > Graph: Added Rule {rule.rule_number} in section '{section_name}'")
            
            # Store General Notes in State (Memory)
            if result.general_notes:
                formatted_notes = f"\n--- PAGE {page_num} NOTES ---\n" + "\n".join(result.general_notes)
                state["general_rules"] += formatted_notes

        except Exception as e:
            logger.exception(f"Failed to parse text rules on page {page_num}: {e}")
            raise
                
    return {"general_rules": state["general_rules"]}



# ---  AGENT 2: PROCESS PLAN ---
def node_process_plans(state: ProjectState,config):
    """
    Processes floor plan pages using minerU-VLM to extract schedules and identify floor plans.
    
    Workflow:
    1. Convert single PDF pages to individual images and isolated PDF files
    2. Apply minerU with VLM backend to separate floor plans from schedule/table information
    3. Perform union cropping to merge titles with their corresponding tables
    4. Extract and analyze each crop: classify as Schedule, Keyed Notes, Plan View, or ignore
    5. Generate vector embeddings using Gemini model for structured data
    6. Store schedule rules and embeddings in Neo4j graph database as GraphRAG nodes
    7. Track and preserve floor plan crops for downstream symbol detection and estimation
    
    Args:
        state (ProjectState): Project state containing:
            - pdf_path: Path to the source PDF
            - output_dir: Directory for temporary outputs
            - page_map: Dictionary mapping page numbers to page types
    
    Returns:
        dict: Updated state with:
            - floor_plan_images: List of paths to identified floor plan crops for later processing
            - general_rules: Status message indicating graph updates
    """
    logger.info("--- NODE: Agent 2 (Plan Ingestion) ---")
    

    floor_plan_images = state.get("floor_plan_images", [])   
    detected_details = state.get("detected_details", [])     
    
    if "remaining_pages" not in state:
        state["remaining_pages"] = [
            p for p, t in state["page_map"].items() if t == "floor"
        ]

    if not state["remaining_pages"]:
        return state
    
    assets_dir = os.getenv("ASSETS_DIR", "/data/assets")
    job_id = config["configurable"]["thread_id"]
    job_dir = os.path.join(assets_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)


    page_num = state["remaining_pages"][0]

    
    logger.debug(f"Ingesting Page {page_num}...")
    
    # 1. Setup Paths
    page_dir = f"{state['output_dir']}/floor_{page_num}"
    page_img_path = f"{page_dir}.png"
    page_pdf_path = f"{page_dir}.pdf"

    mineru_output_dir = f"{state['output_dir']}/floor_{page_num}"
    mineru_vlm_dir = f"{mineru_output_dir}/floor_{page_num}/auto" # Adjust based on actual MinerU output structure
    json_path = f"{mineru_vlm_dir}/floor_{page_num}_content_list_v2.json"
    images_dir = f"{mineru_vlm_dir}/images"

    # 2. Prepare Page Image & PDF
    if not os.path.exists(page_img_path):
        convert_specific_page_to_png(state["pdf_path"], page_num, page_img_path, dpi=300)

    sheet_prefix = state.get("sheet_prefix", "")
    sheet_info =  get_sheet_number(page_img_path, sheet_prefix=sheet_prefix)
    sheet_number = sheet_info["normalized"]
    logger.debug(f"Sheet Number: {sheet_number} processing")
    
    if not os.path.exists(page_pdf_path):
        try:
            reader = PdfReader(state["pdf_path"])
            writer = PdfWriter()
            writer.add_page(reader.pages[page_num])
            with open(page_pdf_path, "wb") as f: writer.write(f)
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")            

    # 3. Run MinerU (Backend)
    if not os.path.exists(json_path):
        logger.debug(f"   > Running MinerU on Page {page_num}...")
        minerU_pdf_creating_extration(page_pdf_path, mineru_output_dir, "pipeline")
    else:
        logger.debug(f"   > MinerU output already exists, skipping | path={json_path}")


    img_orig = cv2.imread(page_img_path)
    img_annotated = img_orig.copy()
    img_h, img_w = img_orig.shape[:2]
    detected_bboxes = []
    mineru_scale_x = 1.0
    mineru_scale_y = 1.0


    if os.path.exists(json_path):
        with open(json_path) as f:
            mineru_data = json.load(f)
        # content_list_v2.json is a list-of-pages; each page is a list of elements
        elements = mineru_data[0] if isinstance(mineru_data, list) else mineru_data

        max_json_x, max_json_y = 0, 0
        for ele in elements:
            bbox = ele.get("bbox")
            if bbox and len(bbox) == 4:
                max_json_x = max(max_json_x, bbox[2])
                max_json_y = max(max_json_y, bbox[3])

        if max_json_x > 0 and max_json_y > 0:
            mineru_scale_x = img_w / max_json_x
            mineru_scale_y = img_h / max_json_y
            logger.debug(f"   > MinerU→PNG scale: x={mineru_scale_x:.3f}, y={mineru_scale_y:.3f}")

        for ele in elements:
            if ele.get("content", {}).get("image_source") or ele.get("type") == "image":
                bbox = ele.get("bbox")
                if bbox and len(bbox) == 4:
                    x1 = int(bbox[0] * mineru_scale_x)
                    y1 = int(bbox[1] * mineru_scale_y)
                    x2 = int(bbox[2] * mineru_scale_x)
                    y2 = int(bbox[3] * mineru_scale_y)
                    cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    detected_bboxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        logger.debug(f"   > Annotated {len(detected_bboxes)} bboxes from MinerU JSON")
    else:
        logger.warning(f"   ! MinerU JSON not found for annotation | path={json_path}")



    annotated_filename = f"page_{page_num}_annotated.png"
    cv2.imwrite(os.path.join(job_dir, annotated_filename), img_annotated)
    cv2.imwrite(os.path.join(job_dir, f"page_{page_num}.png"), img_orig)


    image_url = f"/api/v1/assets/{job_id}/{annotated_filename}"

    state["current_page"] = {
        "page_num": page_num,
        "image_path": page_img_path,   
        "pdf_path": page_pdf_path,     
        "json_path": json_path,
        "sheet_number": sheet_number, 

        "detected_bboxes": detected_bboxes,
        "corrected_bboxes": [],

        "status": "waiting_for_hitl"
    }
    total_floor = len([p for p, t in state["page_map"].items() if t == "floor"])
    remaining_floor = len(state.get("remaining_pages", []))

    review_data = {
        "type": "bbox_review",
        "image_path": image_url,
        "page_num": page_num,
        "bboxes": detected_bboxes,
        "image_width": img_w,
        "image_height": img_h,
        "current_hitl_index": total_floor - remaining_floor + 1,
        "total_hitl_pages": total_floor,
        "remaining_after_this": remaining_floor - 1,
    }

    resume_value = interrupt(review_data)
    if resume_value is None:
            return state
    # corrected_bboxes = resume_value.get("corrected_bboxes", detected_bboxes) if resume_value else detected_bboxes
    state["remaining_pages"].pop(0) 
    corrected_bboxes = resume_value.get("corrected_bboxes", detected_bboxes)
    deleted_mineru_bboxes = resume_value.get("deleted_mineru_bboxes", []) 
    state["current_page"]["corrected_bboxes"] = corrected_bboxes
    state["current_page"]["status"] = "resumed"

    corrected_bboxes = state["current_page"]["corrected_bboxes"]
    logger.info(f"▶️ RESUMED | corrected_bboxes count={len(corrected_bboxes)}")

    logger.info(f"[HITL APPLY] page={page_num} | boxes={len(corrected_bboxes)}")
    if corrected_bboxes:
        os.makedirs(images_dir, exist_ok=True)
        for i, bbox in enumerate(corrected_bboxes):
            x1 = int(bbox["x1"])
            y1 = int(bbox["y1"])
            x2 = int(bbox["x2"])
            y2 = int(bbox["y2"])
            # Guard against out-of-bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"   ! Skipping invalid bbox {bbox}")
                continue
            crop = img_orig[y1:y2, x1:x2]
            crop_filename = f"hitl_crop_{page_num}_{i}.png"
            crop_path = os.path.join(images_dir, crop_filename)
            cv2.imwrite(crop_path, crop)
            logger.debug(f"   > Saved crop: {crop_filename} | bbox=({x1},{y1},{x2},{y2})")
        logger.info(f"   > Saved {len(corrected_bboxes)} user-corrected crops to {images_dir}")
    
    if corrected_bboxes and os.path.exists(json_path):
        with open(json_path, 'r') as f:
            existing_json = json.load(f)
        # v2 json is [[...]] nested
        if isinstance(existing_json, list) and len(existing_json) > 0 and isinstance(existing_json[0], list):
            json_list = existing_json[0]
        else:
            json_list = existing_json

        for i, bbox in enumerate(corrected_bboxes):
            crop_filename = f"hitl_crop_{page_num}_{i}.png"
            crop_full_path = os.path.join(images_dir, crop_filename)
            if os.path.exists(crop_full_path):
                json_list.append({
                    "type": "image",
                    "img_path": f"images/{crop_filename}",
                    "image_caption": [],
                    "image_footnote": [],
                    "bbox": [
                        int(bbox["x1"] / mineru_scale_x),
                        int(bbox["y1"] / mineru_scale_y),
                        int(bbox["x2"] / mineru_scale_x),
                        int(bbox["y2"] / mineru_scale_y)
                    ],
                    "page_idx": 0,
                    "hitl": True
                })

        hitl_bboxes_mineru = [
        [
            int(bbox["x1"] / mineru_scale_x),
            int(bbox["y1"] / mineru_scale_y),
            int(bbox["x2"] / mineru_scale_x),
            int(bbox["y2"] / mineru_scale_y)
        ]
        for bbox in corrected_bboxes
        ]
        deleted_bboxes_mineru = [
            [
                int(bbox["x1"] / mineru_scale_x),
                int(bbox["y1"] / mineru_scale_y),
                int(bbox["x2"] / mineru_scale_x),
                int(bbox["y2"] / mineru_scale_y)
            ]
            for bbox in deleted_mineru_bboxes
        ]

        cleaned_json_list = []

        for item in json_list:
            if item.get("type") == "image" and not item.get("hitl"):
                item_bbox = item.get("bbox", [0, 0, 0, 0])
                explicitly_deleted = any(
                    _bbox_overlap_ratio(item_bbox, dbox) > 0.3
                    for dbox in deleted_bboxes_mineru
                )
                replaced_by_hitl = any(
                    _bbox_overlap_ratio(item_bbox, hbox) > 0.3
                    for hbox in hitl_bboxes_mineru
                )
                if explicitly_deleted or replaced_by_hitl:
                    img_path_in_json = item.get("content", {}).get("image_source", {}).get("path", "")
                    full_img_path = os.path.join(images_dir, os.path.basename(img_path_in_json))
                    if os.path.exists(full_img_path):
                        os.remove(full_img_path)
                        reason = "explicitly deleted by user" if explicitly_deleted else "replaced by HITL crop"
                        logger.debug(f"Removed MinerU image ({reason}): {full_img_path}")
                else:
                    cleaned_json_list.append(item)
            else:
                cleaned_json_list.append(item)
        json_list = cleaned_json_list
        hitl_json_path = json_path.replace("_content_list_v2.json", "_content_list_v2_hitl.json")
        was_nested = isinstance(existing_json, list) and len(existing_json) > 0 and isinstance(existing_json[0], list)
        with open(hitl_json_path, 'w') as f:
            json.dump([json_list] if was_nested else json_list, f)
        active_json_path = hitl_json_path
    else:
        active_json_path = json_path
    
    # 4. Run Union Cropping (Title + Table Merge)
    if os.path.exists(json_path):
        logger.debug(f"   > Running Union Cropping...")
        # This creates UNION crops in 'images_dir' and deletes old ones
        crop_union_tables(active_json_path, page_img_path, output_dir=images_dir)
    else:
        logger.error(f"   ! MinerU JSON not found at {active_json_path}. Skipping Union Crop.")
    
    title_map = {} 
    if os.path.exists(active_json_path) and os.path.exists(page_pdf_path):
        logger.debug("   > Running map_page_layout to associate titles with crops...")
        try:
            layout_groups = map_page_layout(page_pdf_path, active_json_path, images_dir)
            for group in layout_groups:
                    original = group.detail_id
                    group.detail_id = normalize_detail_key(group.detail_id,sheet_number=sheet_number)
                    group.detail_id = normalize_detail_id(group.detail_id)
                    if group.detail_id != original:
                        logger.warning(f"Normalized detail_id: {original} → {group.detail_id}")
            for group in layout_groups:
                for img_file in group.image_files:
                    fname = os.path.basename(img_file)
                    title_map[fname] = group
            logger.debug(f"   > Title map built | {len(title_map)} crop→group mappings")
        except Exception as e:
            logger.warning(f"   ! map_page_layout failed, continuing without title map | error={e}")
    

    # 5. PROCESS CROPS (Ingest Schedules, Identify Plans)
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
        logger.debug(f"   > Found {len(image_files)} crops to analyze.")

        prompt =prompt_for_node_process_plans()
        for img_file in image_files:
            crop_path = os.path.join(images_dir, img_file)

            matched_group = title_map.get(img_file)
            group_title = matched_group.title if matched_group else None
            group_text = "\n".join(matched_group.text_blocks) if matched_group and matched_group.text_blocks else ""
            
            # Prompt: Classify & Extract
            
            msg = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{load_image_base64(crop_path)}"}},
                *(
                    [{"type": "text", "text": f"Known title for this crop: {group_title}\nAssociated text:\n{group_text}"}]
                    if group_title else []
                ),
            ])
            
            try:
                result = llm_flash.with_structured_output(IngestionOutput).invoke([msg])
                resolved_title = group_title or result.title or img_file
                # CASE A: It is a Schedule/Note -> Store in Graph
                if result.type == "Schedule":
                    logger.debug(f"     > Ingested Schedule: {result.title}")
                    rows = result.rows or []
                    logger.debug(f" > Found {len(rows)} rows.")

                    columns = result.columns or []

                    for row in rows:
                        # row is already a dictionary
                        row_data = row

                        # Determine primary key (usually first column)
                        primary_key = None
                        if columns:
                            primary_key = row_data.get(columns[0])
                            if primary_key:
                                primary_key = primary_key.strip().replace("\n", "").replace("\r", "")
                                primary_key = " ".join(primary_key.split())

                        if not primary_key:
                            primary_key = row_data.get("MARK") or row_data.get("KEY") or "UNKNOWN"

                        if primary_key == "UNKNOWN":
                            logger.warning(f"     ! Could not determine primary key for row: {row_data}")

                        graph_db.add_schedule_rule(
                            project_id=config["configurable"]["thread_id"],
                            schedule_name=resolved_title,
                            symbol=primary_key,
                            row_data=row_data,
                            columns=columns,
                            page_num=page_num,
                            sheet_number=sheet_number
                        )

                # CASE B: It is a Plan View -> Save for Agent 5
                elif result.type == "Plan_View":
                    logger.debug(f"     > Found Floor Plan Crop: {img_file}")
                    floor_plan_images.append({
                        "path": crop_path,
                        "sheet": sheet_number,
                        "title": resolved_title, 
                    })
                elif result.type.strip() == "Detail":
                    logger.debug(f"     > Found Detail Crop: {img_file}")

                    if matched_group:
                        detail_id = matched_group.detail_id
                        key = normalize_detail_key(detail_id, sheet_number)
                    else:
                        detail_id = resolved_title
                        key = f"{detail_id}/{sheet_number}" if "/" not in detail_id else detail_id

                    logger.debug(f"     > Detail: {img_file} | detail_id={detail_id} | key={key}")
                    detected_details.append({
                        "detail_id": detail_id,
                        "detail_key": key,
                        "crop_path": crop_path,
                        "sheet": sheet_number,
                        "page": page_num,
                        "title": resolved_title,      
                        "text_blocks": matched_group.text_blocks if matched_group else [],  
                    })

                
            except Exception as e:
                logger.exception(f"     ! Failed to ingest {img_file}: {e}")
    state["current_page"] = None
    return {
        "floor_plan_images": floor_plan_images, 
         "detected_details": detected_details,
        "general_rules": "Updated Graph with Schedules",
        "remaining_pages": state.get("remaining_pages", []),
        "current_page": None            
    }

# ---  AGENT 3: DETAIL PROCESSOR ---
def node_process_details(state: ProjectState,config):
    """
    Extracts and processes section detail drawings using minerU to build a detail
    library and populate the graph database with detail BOMs.

    Detailed steps:
    1. For each page identified as a "section" page, create a single-page PDF and
       corresponding high‑resolution image.
    2. Run minerU in "pipeline" mode on the isolated PDF to perform semantic
       segmentation of title blocks, figures, tables, and other drawing elements.
    3. Provide minerU's JSON output along with the page image to a layout mapper
       that associates titles with their linked figure/table images.
    4. Iterate over the resulting detail groups, extracting structured information
       (e.g. part numbers, materials, dimensions) from each figure or crop.
    5. Capture the sheet number from the drawing for reference and cross‑linking.
    6. Save each detail's data into a local `detail_library` and insert a record into
       Neo4j using `add_detail_bom`, embedding the information so it can be used in
       later semantic searches. Detail keys use engineering notation (e.g., "3/S-3.4").

    Args:
        state (ProjectState): Current project state containing PDF path,
            output directory, and previous processing results.

    Returns:
        dict: Updated state including the populated `detail_library`.
    """
    logger.info("--- NODE 3 : Processing Section Details (MinerU + HITL) ---")
    detail_library = state.get("detail_library", {})
    temp_plan_like_details = state.get("temp_plan_like_details", [])
    temp_dependent_detail_images = state.get("temp_dependent_details", [])

    assets_dir = os.getenv("ASSETS_DIR", "/data/assets")
    job_id = config["configurable"]["thread_id"]
    job_dir = os.path.join(assets_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)

    if "remaining_section_pages" not in state:
        state["remaining_section_pages"] = [
            p for p, t in state["page_map"].items() if t == "section"
        ]

    page_num = state["remaining_section_pages"][0]
    logger.debug(f"Processing Section Page {page_num}...")

    page_img_path = f"{state['output_dir']}/section_page_{page_num}.png"
    page_pdf_path = f"{state['output_dir']}/section_page_{page_num}.pdf"
    mineru_base_dir = f"{state['output_dir']}/section_page_{page_num}/auto"
    layout_pdf_path = f"{mineru_base_dir}/section_page_{page_num}_layout.pdf"
    json_path = f"{mineru_base_dir}/section_page_{page_num}_content_list.json"
    images_dir = f"{mineru_base_dir}/images"
    
    # 2. Extract Single Page PDF & Image
    if not os.path.exists(page_img_path):
        convert_specific_page_to_png(state["pdf_path"], page_num, page_img_path, dpi=300)
    
    try:
        reader = PdfReader(state["pdf_path"])
        writer = PdfWriter()
        writer.add_page(reader.pages[page_num])
        with open(page_pdf_path, "wb") as f: writer.write(f)
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        

    # 3. Extract Sheet Number
    sheet_prefix = state.get("sheet_prefix", "")
    sheet_info = get_sheet_number(page_img_path,sheet_prefix=sheet_prefix)
    sheet_number = sheet_info["normalized"]
    logger.debug(f"   > Identified Sheet Number: {sheet_number}")

    # 4. Run MinerU
    if not os.path.exists(json_path):
        logger.debug(f"   > Running MinerU on Page {page_num}...")
        minerU_pdf_creating_extration(page_pdf_path, state["output_dir"], "pipeline")
    else:
        logger.debug(f"   > MinerU output exists, skipping")
    
    img_orig      = cv2.imread(page_img_path)
    img_annotated = img_orig.copy()
    img_h, img_w  = img_orig.shape[:2]
    detected_bboxes = []
    scale_x = 1.0
    scale_y = 1.0

    if os.path.exists(json_path):
        with open(json_path) as f:
            mineru_data = json.load(f)
        if isinstance(mineru_data, list) and len(mineru_data) > 0 and isinstance(mineru_data[0], list):
                    elements = mineru_data[0]
        else:
            elements = mineru_data 

        max_json_x, max_json_y = 0, 0
        for ele in elements:
            bbox = ele.get("bbox")
            if bbox and len(bbox) == 4:
                max_json_x = max(max_json_x, bbox[2])
                max_json_y = max(max_json_y, bbox[3])

        scale_x = (img_w / max_json_x) if max_json_x > 0 else 1.0
        scale_y = (img_h / max_json_y) if max_json_y > 0 else 1.0
        for ele in elements:
            if ele.get("content", {}).get("image_source") or ele.get("type") == "image":
                bbox = ele.get("bbox")
                if bbox and len(bbox) == 4:
                    x1 = int(bbox[0] * scale_x)
                    y1 = int(bbox[1] * scale_y)
                    x2 = int(bbox[2] * scale_x)
                    y2 = int(bbox[3] * scale_y)
                    cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    detected_bboxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

        annotated_filename = f"section_{page_num}_annotated.png"
        cv2.imwrite(os.path.join(job_dir, annotated_filename), img_annotated)
        cv2.imwrite(os.path.join(job_dir, f"section_{page_num}.png"), img_orig)
        image_url = f"/api/v1/assets/{job_id}/{annotated_filename}"

        state["current_section_page"] = {
        "page_num":       page_num,
        "image_path":     page_img_path,
        "pdf_path":       page_pdf_path,
        "json_path":      json_path,
        "layout_pdf_path": layout_pdf_path,
        "sheet_number":   sheet_number,
        "status": "waiting_for_hitl"
    }
        total_section = len([p for p, t in state["page_map"].items() if t == "section"])
        remaining_section = len(state.get("remaining_section_pages", []))
        review_data = {
            "type": "section_review",
            "image_path":   image_url,
            "page_num":     page_num,
            "bboxes":       detected_bboxes,
            "image_width":  img_w,
            "image_height": img_h,
            "current_hitl_index": total_section - remaining_section + 1,
            "total_hitl_pages": total_section,
            "remaining_after_this": remaining_section - 1,
        }
        resume_value = interrupt(review_data)
        state["remaining_section_pages"].pop(0) 
        corrected_bboxes = resume_value.get("corrected_bboxes", detected_bboxes) if resume_value else detected_bboxes
        deleted_mineru_bboxes = resume_value.get("deleted_mineru_bboxes", []) 
        state["current_section_page"]["corrected_bboxes"] = corrected_bboxes
        state["current_section_page"]["status"] = "resumed"
        logger.info(f"▶️ RESUMED | corrected_bboxes count={len(corrected_bboxes)}")
    # else:

    #     corrected_bboxes = None

    # --- Save corrected crops (same as Agent 2) ---
    if corrected_bboxes:
        os.makedirs(images_dir, exist_ok=True)

        for i, bbox in enumerate(corrected_bboxes):
            x1, y1 = max(0, int(bbox["x1"])), max(0, int(bbox["y1"]))
            x2, y2 = min(img_w, int(bbox["x2"])), min(img_h, int(bbox["y2"]))
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"   ! Skipping invalid bbox {bbox}")
                continue
            crop = img_orig[y1:y2, x1:x2]
            crop_filename = f"section_hitl_crop_{page_num}_{i}.png"
            cv2.imwrite(os.path.join(images_dir, crop_filename), crop)

    if corrected_bboxes and os.path.exists(json_path):
        with open(json_path, 'r') as f:
            existing_json = json.load(f)        
        if isinstance(existing_json, list) and len(existing_json) > 0 and isinstance(existing_json[0], list):
            json_list = existing_json[0]
        else:
            json_list = existing_json

        for i, bbox in enumerate(corrected_bboxes):
            crop_filename = f"section_hitl_crop_{page_num}_{i}.png"
            crop_full_path = os.path.join(images_dir, crop_filename)
            if os.path.exists(crop_full_path):
                json_list.append({
                    "type": "image",
                    "img_path": f"images/{crop_filename}",
                    "image_caption": [],
                    "image_footnote": [],
                    "bbox": [
                        int(bbox["x1"] / scale_x),
                        int(bbox["y1"] / scale_y),
                        int(bbox["x2"] / scale_x),
                        int(bbox["y2"] / scale_y)
                    ],
                    "page_idx": 0,
                    "hitl": True
                })
        hitl_bboxes_mineru = [
        [
            int(bbox["x1"] / scale_x),
            int(bbox["y1"] / scale_y),
            int(bbox["x2"] / scale_x),
            int(bbox["y2"] / scale_y)
        ]
        for bbox in corrected_bboxes
        ]
        deleted_bboxes_mineru = [
            [
                int(bbox["x1"] / scale_x),
                int(bbox["y1"] / scale_y),
                int(bbox["x2"] / scale_x),
                int(bbox["y2"] / scale_y)
            ]
            for bbox in deleted_mineru_bboxes
        ]
        cleaned_json_list = []
        for item in json_list:
            if item.get("type") == "image" and not item.get("hitl"):
                item_bbox = item.get("bbox", [0, 0, 0, 0])
                explicitly_deleted = any(
                    _bbox_overlap_ratio(item_bbox, dbox) > 0.3
                    for dbox in deleted_bboxes_mineru
                )
                replaced_by_hitl = any(
                    _bbox_overlap_ratio(item_bbox, hbox) > 0.3
                    for hbox in hitl_bboxes_mineru
                )
                if explicitly_deleted or replaced_by_hitl:
                    img_path_in_json = item.get("img_path", "")
                    full_img_path = os.path.join(images_dir, os.path.basename(img_path_in_json))
                    if os.path.exists(full_img_path):
                        os.remove(full_img_path)
                        reason = "explicitly deleted by user" if explicitly_deleted else "replaced by HITL crop"
                        logger.debug(f"Removed MinerU image ({reason}): {full_img_path}")
                else:
                    cleaned_json_list.append(item)
            else:
                cleaned_json_list.append(item)
        json_list = cleaned_json_list

        hitl_json_path = json_path.replace("_content_list.json", "_content_list_hitl.json")
        with open(hitl_json_path, 'w') as f:
            json.dump(json_list, f)
        active_json_path = hitl_json_path
    else:
        active_json_path = json_path        
    
    
    
     # --- map_page_layout + extract (for section pages) ---
    if corrected_bboxes is not None:
        if not os.path.exists(layout_pdf_path) or not os.path.exists(json_path):
            logger.warning(f"MinerU output missing for page {page_num}.")
        else:
            logger.debug("   > Running map_page_layout...")
            detail_groups = map_page_layout(layout_pdf_path, active_json_path, images_dir)
            for group in detail_groups:
                original = group.detail_id
                group.detail_id = normalize_detail_key(group.detail_id,sheet_number=sheet_number)
                group.detail_id = normalize_detail_id(group.detail_id)
                if group.detail_id != original:
                    logger.warning(f"Normalized detail_id: {original} → {group.detail_id}")

            if detail_groups:
                logger.debug(f"   > Extracting {len(detail_groups)} details from section page...")
                for group in detail_groups:
                    logger.info(f"   > Extracting: {group.detail_id}")
                    detail_data = extract_single_detail(
                        group,
                        images_dir,
                        temp_plan_like_details,
                        temp_dependent_detail_images,
                        sheet_number,
                        page_num
                    )

                    if detail_data:
                        key = f"{group.detail_id}/{sheet_number}" if "/" not in group.detail_id else group.detail_id
                        detail_library[key] = {
                            "sheet": sheet_number,
                            "page":  page_num,
                            "data":  detail_data.model_dump()
                        }
                        detail_dict = detail_data.model_dump()
                        graph_db.add_detail_bom(
                            project_id=config["configurable"]["thread_id"],
                            detail_key=key,
                            title=detail_dict["title"],
                            materials_list=detail_dict["materials"],
                            fabrication=detail_dict["fabrication"],
                            page_num=page_num,
                            sheet_number=sheet_number
                        )
        
    remaining = state.get("remaining_section_pages", [])
    is_last_section_page = len(remaining) == 0
    if is_last_section_page:
        detected_details = state.get("detected_details", [])
        logger.info(f"--- Processing {len(detected_details)} detected floor-plan details from Agent 2 ---")

        for det in detected_details:
            crop_path    = det["crop_path"]
            sheet_number = det["sheet"]
            page_num     = det["page"]
            detail_id    = det["detail_id"]
            detail_key   = det["detail_key"]

            if not os.path.exists(crop_path):
                logger.warning(f"   ! Crop not found: {crop_path}, skipping")
                continue

            # Classify the crop: is it a standalone Detail or a dependent one?
            img_type = classify_image_as_plan(crop_path)
            logger.debug(f"   > {detail_id} classified as: {img_type}")

            if img_type == "IGNORE":
                logger.debug(f"   > Skipping reference tag: {detail_id}")
                continue

            elif img_type == "DEPENDENT_DETAIL":
                logger.debug(f"   > Queuing as dependent_detail: {detail_id}")
                temp_dependent_detail_images.append({
                    "detail_id": detail_id,
                    "sheet":     sheet_number,
                    "image_path": [crop_path],
                    "page":      page_num,
                    "title":     det.get("title", detail_id),
                })
 
            elif img_type in ("INDEPENDENT_DETAIL", "DETAIL"):
                # Build a minimal DetailGroup to reuse extract_single_detail
                from src.workflow.common.schemas import DetailGroup as DG
                group = DG (
                    detail_id=detail_id,
                    title=det.get("title", detail_id),
                    image_files=[crop_path],
                    text_blocks=det.get("text_blocks", []),
                )

                detail_data = extract_single_detail(
                    group,
                    os.path.dirname(crop_path),
                    temp_plan_like_details=[],   # Agent 2 already handled plan-likes
                    temp_dependent_detail_images=temp_dependent_detail_images,
                    sheet_number=sheet_number,
                    page_num=page_num
                )

                if detail_data:
                    detail_library[detail_key] = {
                        "sheet": sheet_number,
                        "page":  page_num,
                        "data":  detail_data.model_dump()
                    }
                    detail_dict = detail_data.model_dump()
                    graph_db.add_detail_bom(
                        project_id=config["configurable"]["thread_id"],
                        detail_key=detail_key,
                        title=detail_dict["title"],
                        materials_list=detail_dict["materials"],
                        fabrication=detail_dict["fabrication"],
                        page_num=page_num,
                        sheet_number=sheet_number
                    )

            else:
                # PLAN_VIEW from Agent 2's detected_details — shouldn't happen but log it
                logger.warning(f"   ! {detail_id} classified as PLAN_VIEW inside detected_details — skipping")
        
        # ---------------------------------------------------------------
        # PART C: Process temp_dependent_detail_images (collected from both A and B)
        # Same enriched-symbol logic as before
        # ---------------------------------------------------------------
        _process_dependent_details(temp_dependent_detail_images, detail_library,sheet_number, config, state)
        _process_plan_like_details(temp_plan_like_details, detail_library,sheet_number, config, state)
        # temp_dependent_detail_images = []
        # temp_plan_like_details = []


    state["current_section_page"] = None
    return {
    "detail_library": detail_library,
    "remaining_section_pages": state.get("remaining_section_pages", []), 
    "current_section_page": None,   
    "temp_dependent_details": temp_dependent_detail_images,    # ← add
    "temp_plan_like_details": temp_plan_like_details,          # ← add             
}


# ---  AGENT 4: DETAIL PROCESSOR --- 
def node_agent_4_merger(state: ProjectState,config):   
    """
    Merges vision-derived symbols with graph data to drive the final estimation step.

    This node processes each floor plan crop previously identified in
    `floor_plan_images` and performs the following sequence:

    1. Run DINOv2-based object detection on the floor plan to locate symbols.
    2. Crop each detected symbol region and send the crop to Groq for OCR/
       recognition, obtaining a textual symbol identifier.
    3. Use the recognized symbol text to perform a semantic search against the
       Neo4j database, retrieving any associated rule or detail definition.
    4. Augment the symbol information with the retrieved graph data (including
       linked schedules or BOMs) to produce a rich context object.
    5. Invoke the Gemini LLM with the enriched symbols and other metadata to
       generate a pre-fetch estimation output.

    Args:
        state (ProjectState): Current project state containing:
            - pdf_path: Original PDF filename used as project ID
            - floor_plan_images: List of floor plan crop paths produced earlier
    
    Returns:
        dict: A payload containing the results of the final bill of materials
              estimation performed downstream.
    """
    logger.info("--- NODE: Agent 4 (Pre-Fetch Vision Estimator) ---")
    
    project_id = config["configurable"]["thread_id"]
    floor_images = state.get("floor_plan_images", [])
    
    # Load Excel Options
    excel_path = os.getenv("EXCEL_PATH", "Steel Estimator.xlsx")
    valid_materials = get_valid_materials_list(excel_path)
    material_lookup = load_material_weights(excel_path)
    valid_materials_str = json.dumps(valid_materials)

    if not floor_images:
        return {"final_bill_of_materials": {"error": "No floor plans found."}}

    all_extracted_items = []
    job_id = config["configurable"]["thread_id"]
    failed=False
    for img in floor_images:
        img_path = img["path"]
        sheet_number = img["sheet"]
        if not os.path.exists(img_path): 
            logger.warning(f"[Image {img_path}] File not found, skipping")
            continue
        
        filename = os.path.basename(img_path)
        logger.debug(f"  > Processing Image: {filename}")

        # 1. Run Symbol Detection (DINO + Groq)
        symbol_out_dir = os.path.join(os.path.dirname(img_path), "detected_symbols")
        try:
            raw_symbols = detect_and_read_symbols(img_path, symbol_out_dir)
            raw_symbols = [s for s in raw_symbols if s.get("text_content") != "Unknown"]
            logger.debug(f"Here is the Raw symbol we detectd : {raw_symbols}")
        except Exception as e:
            logger.error(f"    ! Symbol detection failed: {e}")
            raw_symbols = []

        # 2. PRE-FETCH DEFINITIONS (The Fix)
    
        enriched_symbols = []
        for sym in raw_symbols:
            logger.info(f"RAW SYMBOL : {sym}")
            query_text = sym.get("text_content", "").strip()
            logger.info(f"query text : {query_text}")
            query_text = query_text.upper()
            
            definition = None

            if is_detail_ref(query_text):
                logger.info(f"Using DIRECT LOOKUP for {query_text}")

                definition = graph_db.get_definition_by_id(
                    query_text,
                    project_id
                )

            else:
                logger.info(f"Using SEMANTIC SEARCH for {query_text}")

                matches = graph_db.semantic_search(
                    query_text,
                    project_id,
                    sheet_number=None,
                    limit=3
                )

                if matches:
                    definition = matches[0]



            if definition:
                logger.info(f"DEFINITION FOUND: {definition}")

                if definition.get("BOM") is not None:
                    for item in definition["BOM"]:
                        logger.info(f"Item : {item}")

                        rule_text = item.get("qty_rule", "")
                        mat_text = item.get("item_name", "") or ""

                        logger.info(f"Material text used: {mat_text}")

                        schedule_keywords = ["schedule", "see plan", "see sched", "per schedule"]
                        text_blob = f"{rule_text} {mat_text}".lower()

                        if any(k in text_blob for k in schedule_keywords) and mat_text:
                            logger.debug(f"    > Resolving Reference: {mat_text}")

                            sub_matches = graph_db.semantic_search(
                                mat_text,
                                project_id,
                                sheet_number=None,
                                limit=3
                            )

                            if sub_matches:
                                schedule_obj = sub_matches[0]

                                item["linked_schedule_data"] = {
                                    "schedule_id": schedule_obj.get("ID"),
                                    "schedule_name": schedule_obj.get("Name"),
                                    "columns": schedule_obj.get("Columns"),
                                    "rows": schedule_obj.get("Rows"),
                                    "sheet": schedule_obj.get("Sheet")
                                }

                                logger.debug(f"      -> Found schedule: {schedule_obj.get('ID')}")

            # Attach the fully enriched definition to the symbol
            sym['linked_definition'] = definition
            enriched_symbols.append(sym)
        logger.debug(f"    > Enriched {len(enriched_symbols)} symbols with Graph Data.")

        # 3. ONE-SHOT PROMPT (No ReAct Loop needed anymore!
        system_prompt =prompt_for_agent_4_merger(json.dumps(enriched_symbols, indent=2),valid_materials_str,sheet_number)

        # 4. Call LLM (Standard Invoke)
        b64 = load_image_base64(img_path)
        msg = HumanMessage(content=[
            {"type": "text", "text": system_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        ])
        
        try:
            result = llm_pro.with_structured_output(FinalEstimation).invoke([msg])
            
            if result.final_bill_of_materials:
                logger.debug(f"    > Extracted {len(result.final_bill_of_materials)} items.")
                all_extracted_items.extend(result.final_bill_of_materials)
                  
        except Exception as e:
            logger.error(f"Agent 4 failed: {e}")
            
            failed=True
    enriched = []
    if all_extracted_items:
        bom_dicts = [item.model_dump() for item in all_extracted_items]
        enriched = enrich_bom_with_pricing(bom_dicts, material_lookup)
    valid_set = {m.upper().strip() for m in valid_materials}

    filtered = []
    for item in all_extracted_items:
        size = (item.material_size or "").upper().strip()
        
        # Drop only the truly bad ones
        if not size or size == "UNKNOWN":
            logger.warning(f"Dropping BOM item with unknown size: {item.description}")
            continue
        
        # If exact match → keep as-is
        if size in valid_set:
            filtered.append(item)
            continue
        
        # If no exact match → keep but flag for review
        logger.warning(f"BOM item not in valid materials list: {item.description} ({size}) — keeping but flagged")
        filtered.append(item)

    all_extracted_items = filtered

    

    if failed :
        update_job_status(job_id, "failed")
    else:
        update_job_status(job_id, "completed")
        update_job_progress(job_id, "completed", "agent_4_merger")
        # save the BOM in .json file   
        job_id = config["configurable"]["thread_id"]
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../../"))
        base_path = os.getenv("BOM_STORAGE_PATH",  os.path.join(PROJECT_ROOT, "bom_storage"))
        os.makedirs(base_path, exist_ok=True)
        file_path = os.path.join(base_path, f"{job_id}.json")
        data = {
            "job_id": job_id,
            "bom": enriched
        }
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"BOM saved successfully at {file_path}")
        except Exception as e:
            logger.error(f"Failed to save BOM | job_id={job_id} | error={str(e)}")


    return {"final_bill_of_materials": {"final_bill_of_materials": enriched}}   

