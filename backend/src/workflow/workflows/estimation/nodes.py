import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
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
    prompt_for_agent_4_merger
    )
from src.workflow.common.utils import (
    crop_union_tables,
    map_page_layout,
    extract_single_detail,
    get_valid_materials_list,
    load_image_base64,
    get_sheet_number,
    convert_specific_page_to_png,
    load_material_weights,              
    minerU_pdf_creating_extration                
    )

from src.infrastructure.graph_db import graph_db
from src.infrastructure.symbol_detection import detect_and_read_symbols 
from src.logger import setup_logger
from src.db.update_jobs_status import update_job_status

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
    logger.info("--- NODE 1 : Classifying Pages ---")
    pdf_path = state["pdf_path"]
    output_dir=state['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    page_map = {}
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        
    for page_num in range(total_pages):
        temp_img_path = f"{state['output_dir']}/temp_page_{page_num}.png"
        convert_specific_page_to_png(pdf_path, page_num, temp_img_path, dpi=150)

        prompt=prompt_for_node_classify_pages()

        image_b64 = load_image_base64(temp_img_path)
        msg = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        ])

        result = llm_flash.with_structured_output(DrawingTypeResponse).invoke([msg])
        page_map[page_num] = result.drawing_type
        logger.info(f"Page Index {page_num}: {result.drawing_type}")

    return {"page_map": page_map}


# ---  AGENT 1: TEXT PROCESSOR ---
def node_process_text_rules(state: ProjectState):
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
    logger.info("--- NODE: Processing Text Rules ---")
    text_pages = [p for p, t in state["page_map"].items() if t == "text"]
    
    for page_num in text_pages:
        # 1. Extract Single Page PDF
        page_pdf_path = f"{state['output_dir']}/notes_{page_num}.pdf"
        try:
            reader = PdfReader(state["pdf_path"])
            writer = PdfWriter()
            writer.add_page(reader.pages[page_num])
            with open(page_pdf_path, "wb") as f: writer.write(f)
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            continue

        # 2. Run MinerU (Assuming this function works and saves to the path below)
        logger.info(f"   > Running MinerU on Page {page_num}...")
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
        logger.info("   > Calling LLM to parse Rules...")
        
        prompt=prompt_node_process_text_rules(markdown_content)

        msg = HumanMessage(content=prompt)
        
        try:
            result = llm_flash.with_structured_output(TextRulesExtraction).invoke([msg])
            
            # Store Rules in Graph
            for rule in result.rules:
                graph_db.add_schedule_rule(
                    project_id=os.path.basename(state["pdf_path"]), # Use filename as ID
                    schedule_name=rule.schedule_name,
                    symbol=rule.symbol,
                    specs=rule.specs,
                    page_num=page_num
                )
                logger.info(f"   > Graph: Added Rule '{rule.symbol}' for {rule.schedule_name}")
            
            # Store General Notes in State (Memory)
            if result.general_notes:
                formatted_notes = f"\n--- PAGE {page_num} NOTES ---\n" + "\n".join(result.general_notes)
                state["general_rules"] += formatted_notes

        except Exception as e:
            logger.info(f"Failed to parse text rules on page {page_num}: {e}")
                
    return {"general_rules": state["general_rules"]}

# ---  AGENT 2: PROCESS PLAN ---
def node_process_plans(state: ProjectState):
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
    
    # New State Variable to hold the plan images for Agent 5
    floor_plan_images = [] 
    
    floor_pages = [p for p, t in state["page_map"].items() if t == "floor"]
    
    for page_num in floor_pages:
        logger.info(f"Ingesting Page {page_num}...")
        
        # 1. Setup Paths
        page_dir = f"{state['output_dir']}/floor_{page_num}"
        page_img_path = f"{page_dir}.png"
        page_pdf_path = f"{page_dir}.pdf"

        mineru_output_dir = f"{state['output_dir']}/floor_{page_num}"
        mineru_vlm_dir = f"{mineru_output_dir}/floor_{page_num}/vlm" # Adjust based on actual MinerU output structure
        json_path = f"{mineru_vlm_dir}/floor_{page_num}_content_list_v2.json"
        images_dir = f"{mineru_vlm_dir}/images"

        # 2. Prepare Page Image & PDF
        convert_specific_page_to_png(state["pdf_path"], page_num, page_img_path, dpi=300)

        sheet_number = get_sheet_number(page_img_path)
        
        try:
            reader = PdfReader(state["pdf_path"])
            writer = PdfWriter()
            writer.add_page(reader.pages[page_num])
            with open(page_pdf_path, "wb") as f: writer.write(f)
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            continue

        # 3. Run MinerU (VLM Backend)
        logger.info(f"   > Running MinerU (VLM) on Page {page_num}...")
        minerU_pdf_creating_extration(page_pdf_path, mineru_output_dir, "vlm-auto-engine")

        # 4. Run Union Cropping (Title + Table Merge)
        if os.path.exists(json_path):
            logger.info(f"   > Running Union Cropping...")
            # This creates UNION crops in 'images_dir' and deletes old ones
            crop_union_tables(json_path, page_img_path, output_dir=images_dir)
        else:
            logger.error(f"   ! MinerU JSON not found at {json_path}. Skipping Union Crop.")

        # 5. PROCESS CROPS (Ingest Schedules, Identify Plans)
        if os.path.exists(images_dir):
            image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
            logger.info(f"   > Found {len(image_files)} crops to analyze.")
            
            for img_file in image_files:
                crop_path = os.path.join(images_dir, img_file)
                
                # Prompt: Classify & Extract
                prompt =prompt_for_node_process_plans()
                msg = HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{load_image_base64(crop_path)}"}}
                ])
                
                try:
                    result = llm_flash.with_structured_output(IngestionOutput).invoke([msg])
                    
                    # CASE A: It is a Schedule/Note -> Store in Graph
                    if result.type == "Schedule":
                        logger.info(f"     > Ingested Schedule: {result.title}")
                        rows = result.rows or []
                        logger.info(f" > Found {len(rows)} rows.")
                        logger.info(f"     > Found {len(result.rows)} rows.")

                        columns = result.columns or []

                        for row in rows:
                            # row is already a dictionary
                            row_data = row

                            # Determine primary key (usually first column)
                            primary_key = None
                            if columns:
                                primary_key = row_data.get(columns[0])

                            if not primary_key:
                                primary_key = row_data.get("MARK") or row_data.get("KEY") or "UNKNOWN"

                            if primary_key == "UNKNOWN":
                                logger.warning(f"     ! Could not determine primary key for row: {row_data}")

                            graph_db.add_schedule_rule(
                                project_id=os.path.basename(state["pdf_path"]),
                                schedule_name=result.title,
                                symbol=primary_key,
                                row_data=row_data,
                                columns=columns,
                                page_num=page_num,
                                sheet_number=sheet_number
                            )

                    # CASE B: It is a Plan View -> Save for Agent 5
                    elif result.type == "Plan_View":
                        logger.info(f"     > Found Floor Plan Crop: {img_file}")
                        floor_plan_images.append({
                            "path": crop_path,
                            "sheet": sheet_number
                        })
                except Exception as e:
                    logger.error(f"     ! Failed to ingest {img_file}: {e}")
                    
    return {
        "floor_plan_images": floor_plan_images, 
        "general_rules": "Updated Graph with Schedules"
    }

# ---  AGENT 3: DETAIL PROCESSOR ---
def node_process_details(state: ProjectState):
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
    logger.info("--- NODE 3 : Processing Section Details (MinerU) ---")
    detail_library = state.get("detail_library", {})
    section_pages = [p for p, t in state["page_map"].items() if t == "section"]
    logger.info(f"Length of the section page : {len(section_pages)}")
    for page_num in section_pages:
        logger.info(f"Processing Page {page_num}...")
        
        # 1. Setup Paths
        page_img_path = f"{state['output_dir']}/section_page_{page_num}.png"
        page_pdf_path = f"{state['output_dir']}/section_page_{page_num}.pdf"
        
        # 2. Extract Single Page PDF & Image
        convert_specific_page_to_png(state["pdf_path"], page_num, page_img_path, dpi=300)
        
        try:
            reader = PdfReader(state["pdf_path"])
            writer = PdfWriter()
            writer.add_page(reader.pages[page_num])
            with open(page_pdf_path, "wb") as f: writer.write(f)
        except Exception as e:
            logger.info(f"PDF extraction failed: {e}")
            continue

        # 3. Extract Sheet Number
        sheet_number = get_sheet_number(page_img_path)
        logger.info(f"   > Identified Sheet Number: {sheet_number}")

        # 4. Run MinerU 
        logger.info(f"   > Running MinerU on Page {page_num}...")
        minerU_pdf_creating_extration(page_pdf_path, state["output_dir"],"pipeline")

        # 5. Prepare MinerU Data. 
        mineru_base_dir = f"{state['output_dir']}/section_page_{page_num}/auto"
        layout_pdf_path = f"{mineru_base_dir}/section_page_{page_num}_layout.pdf"
        json_path = f"{mineru_base_dir}/section_page_{page_num}_content_list.json"
        images_dir = f"{mineru_base_dir}/images"

        logger.info("   > Step 1: Mapping Layout...")
        if not os.path.exists(layout_pdf_path) or not os.path.exists(json_path):
            logger.warning(
    f"MinerU output missing for page {page_num}. "
    f"layout={layout_pdf_path}, json={json_path}"
)
            continue
        detail_groups = map_page_layout(layout_pdf_path, json_path, images_dir)

        if not detail_groups:
            logger.warning("   ! No details found on page.")
            continue

   
        logger.info(f"   > Step 2: Extracting {len(detail_groups)} details...") 
        
        for group in detail_groups:
                # Call the extractor for this specific group
                logger.info(f" > Extracting detail: {group.detail_id}")
                detail_data = extract_single_detail(group, images_dir)
                
                if detail_data:
                    # Construct Key (Clean logic)
                    key = f"{group.detail_id}/{sheet_number}" if "/" not in group.detail_id else group.detail_id
                    detail_library[key] = {
                    "sheet": sheet_number,
                    "page": page_num,
                    "data": detail_data.model_dump()
                    }
                    
                    graph_db.add_detail_bom(
                        project_id=os.path.basename(state["pdf_path"]),
                        detail_key=key, 
                        title=detail_data.title, 
                        materials_list=detail_data.model_dump()["materials"], 
                        page_num=page_num,
                        sheet_number=sheet_number
                    )

    return {"detail_library": detail_library}


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
    
    project_id = os.path.basename(state["pdf_path"])
    floor_images = state.get("floor_plan_images", [])
    
    # Load Excel Options
    excel_path = "#1A Steel Estimator (2023).xlsx"
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
        if not os.path.exists(img_path): continue
        
        filename = os.path.basename(img_path)
        logger.info(f"  > Processing Image: {filename}")

        # 1. Run Symbol Detection (DINO + Groq)
        symbol_out_dir = os.path.join(os.path.dirname(img_path), "detected_symbols")
        try:
            raw_symbols = detect_and_read_symbols(img_path, symbol_out_dir)
            raw_symbols = [s for s in raw_symbols if s.get("text_content") != "Unknown"]
            logger.info(f"Here is the Raw symbol we detectd : {raw_symbols}")
        except Exception as e:
            logger.error(f"    ! Symbol detection failed: {e}")
            raw_symbols = []

        # 2. PRE-FETCH DEFINITIONS (The Fix)
    
        enriched_symbols = []
        for sym in raw_symbols:
            query_text = f"{sym.get('shape','')} {sym.get('text_content','')}"
            
            # A. Primary Lookup (Symbol -> Detail/Rule)
            matches = graph_db.semantic_search(query_text, project_id,sheet_number=sheet_number,limit=1)
            
            definition = None
            
            if matches and matches[0]['score'] > 0.80:
                definition = matches[0] # Use the full object
                
                # B. Recursive Lookup (Detail Component -> Schedule)
                if definition.get("BOM"):
                    for item in definition["BOM"]:
                        # Check for "Schedule" or "See Plan" in rule/material
                        rule_text = item.get("qty_rule", "")
                        mat_text = item.get("material", "") or ""
                        schedule_keywords = ["schedule", "see plan", "see sched", "per schedule"]
                        text_blob = f"{rule_text} {mat_text}".lower()
                        if any(k in text_blob for k in schedule_keywords):
                            logger.info(f"    > Resolving Reference: {mat_text}")
                            
                            # Search for the Schedule (Filter by Label="Schedule" if possible)
                            # Using the material name as the query (e.g. "LOOSE LINTEL")
                            sub_matches = graph_db.semantic_search(mat_text, project_id,sheet_number=sheet_number, limit=1)
                            
                            if sub_matches:
                                schedule_obj = sub_matches[0]

                                item["linked_schedule_data"] = {
                                    "schedule_id": schedule_obj.get("ID"),
                                    "schedule_name": schedule_obj.get("Name"),
                                    "columns": schedule_obj.get("Columns"),
                                    "rows": schedule_obj.get("Rows"),
                                    "sheet": schedule_obj.get("Sheet")
                                }

                                logger.info(f"      -> Found schedule: {schedule_obj.get('ID')}")

            # Attach the fully enriched definition to the symbol
            sym['linked_definition'] = definition
            enriched_symbols.append(sym)
        logger.info(f"    > Enriched {len(enriched_symbols)} symbols with Graph Data.")

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
                logger.info(f"    > Extracted {len(result.final_bill_of_materials)} items.")
                all_extracted_items.extend(result.final_bill_of_materials)
                  
        except Exception as e:
            logger.error(f"Agent 4 failed: {e}")
            
            failed=True
    if failed :
        update_job_status(job_id, "Failed")
    else:
        update_job_status(job_id, "Completed")

    return {"final_bill_of_materials": {"final_bill_of_materials": [item.model_dump() for item in all_extracted_items]}}   

