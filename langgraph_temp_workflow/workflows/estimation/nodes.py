import os
import json

from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import  Literal, List, Optional, Dict, Any
# LangChain / LangGraph Imports

from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage

# import state from the common/state file 
from langgraph_temp_workflow.common.state import ProjectState

# import schema from the common/schema file 
from langgraph_temp_workflow.common.schemas import DrawingTypeResponse 
from langgraph_temp_workflow.common.schemas import DetailExtraction ,DetailList
from langgraph_temp_workflow.common.schemas import PlanExtraction 
from langgraph_temp_workflow.common.schemas import FinalEstimation 
from langgraph_temp_workflow.common.schemas import TextRulesExtraction 

#import the schemas 

from langgraph_temp_workflow.workflows.estimation.prompt import prompt_for_node_classify_pages
from langgraph_temp_workflow.workflows.estimation.prompt import prompt_for_node_process_details
from langgraph_temp_workflow.workflows.estimation.prompt import prompt_for_node_process_plans
from langgraph_temp_workflow.workflows.estimation.prompt import prompt_for_agent_4_merger

# import the utils  function 
from utils.minerU_pdf_reading import minerU_pdf_creating_extration
from langgraph_temp_workflow.common.utils import crop_union_tables
# from langgraph_temp_workflow.common.utils import extract_detail_components_with_crops

# import tools
from langgraph_temp_workflow.tools.graph_tools import lookup_symbol_definition 
from langgraph_temp_workflow.tools.graph_tools import submit_final_estimate

# import graph DB here

from utils.graph_db import graph_db

# PDF & Image Processing Imports
import pdfplumber
from pypdf import PdfReader, PdfWriter
from utils.pdf_page_to_png import convert_specific_page_to_png

# --- CUSTOM UTILS ---
from utils.croped_sections import crop_sections_from_page
# from utils.sementic_segmentation import semantic_segmentation_app
from langgraph_temp_workflow.workflows.segmentation.graph import semantic_segmentation_app
from langgraph_temp_workflow.common.utils import load_image_base64
from langgraph_temp_workflow.common.utils import extract_text_from_response
from langgraph_temp_workflow.common.utils import get_sheet_number
from langgraph_temp_workflow.common.utils import find_title_coordinates_from_image_and_pdf
from langgraph_temp_workflow.common.utils import preprocess_image_inplace

load_dotenv()


# --- 1. SETUP MODELS ---
llm_pro = ChatGoogleGenerativeAI(model="gemini-3-pro-preview") 
llm_25_pro = ChatGoogleGenerativeAI(model="gemini-2.5-pro") 
llm_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite") 




def node_classify_pages(state: ProjectState):
    print("--- NODE: Classifying Pages ---")
    pdf_path = state["pdf_path"]
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
        print(f"Page Index {page_num}: {result.drawing_type}")

    return {"page_map": page_map}

# --- 5. AGENT 1: TEXT PROCESSOR ---
def node_process_text_rules(state: ProjectState):
    print("--- NODE: Processing Text Rules ---")
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
            print(f"PDF extraction failed: {e}")
            continue

        # 2. Run MinerU (Assuming this function works and saves to the path below)
        print(f"   > Running MinerU on Page {page_num}...")
        minerU_pdf_creating_extration(page_pdf_path, state["output_dir"],"pipeline")

        # 3. Read the Markdown File
        # Note: Adjust path logic if MinerU creates subfolders differently
        md_file_path = f"{state['output_dir']}/notes_{page_num}/auto/notes_{page_num}.md"
        
        try:
            with open(md_file_path, "r", encoding="utf-8") as f:
                markdown_content = f.read()
        except FileNotFoundError:
            print(f"   ! Markdown file not found: {md_file_path}")
            continue

        # 4. The Advanced Prompt
        print("   > Calling LLM to parse Rules...")
        
        prompt = f"""
        You are a Structural Engineer analyzing the "General Notes" and "Schedules" of a construction project.
        The input below is a Markdown file extracted from the PDF.

        ### YOUR GOAL
        Extract structured **Rules** and **Protocols** that will guide the estimation process.

        ### INPUT MARKDOWN:
        {markdown_content}

        ### INSTRUCTIONS:

        **1. PARSE SCHEDULES (Tables):**
        - Look for Markdown tables (lines starting with `|`).
        - Identify the Table Name (e.g., "Shear Wall Schedule", "Lintel Schedule", "Footing Schedule").
        - For each row, extract the **Symbol/Mark** (Column 1) and the **Specifications** (Other Columns).
        - *Example:* If row is `| 1 | 5/8" Bolt | 16" OC |`, create a Rule: `Symbol="1", Specs="5/8" Bolt @ 16" OC"`.

        **2. EXTRACT GENERAL PROTOCOLS (Text):**
        - Look for sections like "STRUCTURAL STEEL", "CONCRETE", "WOOD".
        - Extract **Global Defaults** that affect estimation.
        - *Example:* "All structural steel shall be ASTM A992." -> Keep this.
        - *Example:* "Concrete strength 3000 psi." -> Keep this.
        - *Example:* "Notify architect of discrepancies." -> Ignore (Administrative).

        **3. OUTPUT FORMAT:**
        Return a JSON object matching the `TextRulesExtraction` schema.
        - `rules`: List of specific schedule items.
        - `general_notes`: List of global material specs.
        """
        
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
                print(f"   > Graph: Added Rule '{rule.symbol}' for {rule.schedule_name}")
            
            # Store General Notes in State (Memory)
            if result.general_notes:
                formatted_notes = f"\n--- PAGE {page_num} NOTES ---\n" + "\n".join(result.general_notes)
                state["general_rules"] += formatted_notes

        except Exception as e:
            print(f"Failed to parse text rules on page {page_num}: {e}")
                
    return {"general_rules": state["general_rules"]}

# --- 6. AGENT 3: DETAIL PROCESSOR ---
from langgraph_temp_workflow.common.utils import map_page_layout, extract_single_detail

def node_process_details(state: ProjectState):
    print("--- NODE: Processing Details (MinerU) ---")
    detail_library = state.get("detail_library", {})
    section_pages = [p for p, t in state["page_map"].items() if t == "section"]

    for page_num in section_pages:
        print(f"Processing Page {page_num}...")
        
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
            print(f"PDF extraction failed: {e}")
            continue

        # 3. Extract Sheet Number
        sheet_number = get_sheet_number(page_img_path)
        print(f"   > Identified Sheet Number: {sheet_number}")

        # 4. Run MinerU 
        print(f"   > Running MinerU on Page {page_num}...")
        minerU_pdf_creating_extration(page_pdf_path, state["output_dir"],"pipeline")

        # 5. Prepare MinerU Data. 
        mineru_base_dir = f"{state['output_dir']}/section_page_{page_num}/auto"
        layout_pdf_path = f"{mineru_base_dir}/section_page_{page_num}_layout.pdf"
        json_path = f"{mineru_base_dir}/section_page_{page_num}_content_list.json"
        images_dir = f"{mineru_base_dir}/images"

        print("   > Step 1: Mapping Layout...")
        detail_groups = map_page_layout(layout_pdf_path, json_path, images_dir)

        if not detail_groups:
            print("   ! No details found on page.")
            continue

        if not os.path.exists(layout_pdf_path) or not os.path.exists(json_path):
            print(f"   ! MinerU output missing for page {page_num}. Skipping.")
            continue
       # 2. EXTRACT EACH DETAIL
        print(f"   > Step 2: Extracting {len(detail_groups)} details...") 
        
    #     print("   > Calling Extraction Utility...")
    #     try:
    #         # This function must return a DetailList object
    #         result = extract_detail_components_with_crops(
    #             pdf_layout_path=layout_pdf_path,
    #             json_path=json_path,
    #             images_dir=images_dir
    #         )
    #         # 7. Process Results & Store (Same logic as before)
    #         # We iterate through the list returned by the utility
    #         # 7. Process Results & Store
    #         if result and result.details:
    #             for detail in result.details:
                    
                   
    #                 clean_sheet = sheet_number.replace("Sheet", "").strip()
                    
    #                 det_num = detail.detail_number
    #                 if det_num and str(det_num).lower() != "null":
    #                     clean_num = str(det_num).strip().replace(".", "")
    #                     key = f"{clean_num}/{clean_sheet}"
    #                 else:
    #                     # Fallback: Use Title
    #                     safe_title = "".join(x for x in detail.title if x.isalnum())
    #                     key = f"{safe_title}/{clean_sheet}"
                    
    #                 print(f"   > Stored Detail: {key}")
                    
    #                 # Store in State
    #                 detail_library[key] = detail.model_dump()
                    
    #                 # --- FIX 2: CALL GRAPH DB ---
    #                 # Ensure graph_db.py is saved!
    #                 graph_db.add_detail_bom(
    #                     project_id=os.path.basename(state["pdf_path"]),
    #                     detail_key=key, 
    #                     title=detail.title, 
    #                     materials_list=detail.model_dump()["materials"], 
    #                     page_num=page_num
    #                 )
    #         else:
    #             print(f"   ! No details extracted for page {page_num}")

    #     except Exception as e:
    #         print(f"Failed to process page {page_num}: {e}")

    # return {"detail_library": detail_library}

        for group in detail_groups:
                # Call the extractor for this specific group
                detail_data = extract_single_detail(group, images_dir)
                
                if detail_data:
                    # Construct Key (Clean logic)
                    key = group.detail_id # The Mapper already extracted "7/S-3.2"
                    
                    # Store
                    detail_library[key] = detail_data.model_dump()
                    
                    graph_db.add_detail_bom(
                        project_id=os.path.basename(state["pdf_path"]),
                        detail_key=key, 
                        title=detail_data.title, 
                        materials_list=detail_data.model_dump()["materials"], 
                        page_num=page_num
                    )

    return {"detail_library": detail_library}


# --- . AGENT 2: PLAN ESTIMATOR (RAW DATA COLLECTOR) ---
# Here what we are doing is:
# we call the minearU and collect the layout
# inout we are giving to vlm is that we give the josn file 
# JSON FILE -> wich have the table + image and notes too and. here we can make sure if the notes and header are the same or not. 
# amy be we can convert the notes union and crop and stored them and in case if the image url is present then we will deletec that and stored the new croped one
# second as we got all list of image and now we treverse through each one and give to vlm to see if the current one is table or schedules or floor plan
# if the table or schedule or notes then we need to streod that informeation 
# if the floor or any plan kind of then stored the address in the variable so that in the last stee of agent we will call to process this florr plan 

# def node_process_plans(state: ProjectState):
#     print("--- NODE: Agent 2 (Plan Scanner) ---")
#     raw_plan_data = []
#     current_detail_library = state.get("detail_library", {}).copy()
#     floor_pages = [p for p, t in state["page_map"].items() if t == "floor"]
    
#     for page_num in floor_pages:
#         print(f"Scanning Page Index {page_num}...")
        
#         page_dir = f"{state['output_dir']}/floor_{page_num}"
        
#         page_img_path = f"{state['output_dir']}/floor_{page_num}.png"
#         page_pdf_path = f"{state['output_dir']}/floor_{page_num}.pdf"

#         semantic_crops_dir = f"{state['output_dir']}/floor_{page_num}"

#         # Here created image
#         convert_specific_page_to_png(state["pdf_path"], page_num, page_img_path, dpi=300)

#         # #Here created pdf
#         # try:
#         #     reader = PdfReader(state["pdf_path"])
#         #     writer = PdfWriter()
#         #     writer.add_page(reader.pages[page_num])
#         #     with open(page_pdf_path, "wb") as f: writer.write(f)
#         # except Exception as e:
#         #     print(f"PDF extraction failed: {e}")
#         #     continue
        
#         # child_initial_state = {
#         #     "image_path": page_img_path,
#         #     "pdf_path":page_pdf_path,
#         #     "detected_queue": [], 
#         #     "final_crops": [], 
#         #     "current_retry_count": 0,
#         #     "current_region_label": None, 
#         #     "current_bbox": None,
#         #     "output_dir": state["output_dir"], # Pass the base output dir
#         #     "extracted_data": {}
#         # }
#         # # call to the sementic segementation code to get the focused image 
#         # semantic_segmentation_app.invoke(child_initial_state, config={"recursion_limit": 150})

#         #Here we will call the ml model to do the sementic segmentation
#         # get_coordinates_of_the_segmentation(page_img_path,semantic_crops_dir)

#         # Call the minerU to crop the image into sections

#         print(f"   > Running MinerU on Page {page_num}...")
#         minerU_pdf_creating_extration(page_pdf_path, state["output_dir"])

#         mineru_base_dir = f"{state['output_dir']}/section_page_{page_num}/auto"

#         # to tell the vlm what we have detected from the bigger picture
#         layout_pdf_path = f"{mineru_base_dir}/section_page_{page_num}_layout.pdf"
#         # json file for getting the flow of and see the table and fogires
#         json_path = f"{mineru_base_dir}/section_page_{page_num}_content_list.json"
        
#         #to get the actual floor plan from the list of the images(which content table and figures).
#         images_dir = f"{mineru_base_dir}/images" 


#         if not os.path.exists(layout_pdf_path) or not os.path.exists(json_path):
#             print(f"   ! MinerU output missing for page {page_num}. Skipping.")
#             continue

#         with open(json_path, 'r') as f:
#             json_data = json.load(f)
#             text_data = [item for item in json_data if item["type"] == "text"]
#             json_string = json.dumps(text_data, indent=2)

#         contents_payload = []
#         prompt = prompt_for_node_process_details()

#         with open(layout_pdf_path, "rb") as f:
#             contents_payload.append(types.Part.from_bytes(data=f.read(), mime_type='application/pdf'))

#         if os.path.exists(images_dir):
#             image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
#             print(f"   > Loading {len(image_files)} cropped images...")
#             for img_file in image_files:
#                 img_path = os.path.join(images_dir, img_file)
#                 with open(img_path, "rb") as f:
#                     contents_payload.append(types.Part.from_text(text=f"Image File: {img_file}"))
#                     contents_payload.append(types.Part.from_bytes(data=f.read(), mime_type='image/jpeg'))

#         contents_payload.append(types.Part.from_text(text=f"OCR Text Data:\n{json_string}"))
#         print("   > Sending MinerU data to Gemini...")

#         # it will stores the sementic segemntation image in the folder named as floor/roof
#         sementic_croped_images = [] 
#         global_img_b64 = load_image_base64(page_img_path)
        
#         plan_croped_image:str
        
#         if os.path.exists(semantic_crops_dir):
#             for f in os.listdir(semantic_crops_dir):
#                 if f.endswith(".png"):
#                     img_path = os.path.join(semantic_crops_dir, f)

#                     success = preprocess_image_inplace(img_path)
#                     if success:
#                         sementic_croped_images.append(img_path)
             
#             print(f" > Found {len(sementic_croped_images)-1} semantic crops (processed in-place).")
#         else:
#             print(f" ! Warning: No semantic crops found at {semantic_crops_dir}")
        
#         # --- ADVANCED PROMPT FOR AGENT 2 ---
#         prompt=prompt_for_node_process_plans()

#         for q in sementic_croped_images:
#             content = [
#                 {"type": "text", "text": prompt},
#                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{global_img_b64}"}},
#                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{load_image_base64(q)}"}}
#             ]
            
#             msg = HumanMessage(content=content)
            
#             try:
#                 result = llm_pro.with_structured_output(PlanExtraction).invoke([msg])
#                 data = result.model_dump()

#                 #--- PUSH TO NEO4J ----
#                 #1. Push Members (Beams/Cols)
                
#                 for m in data["members"]:
#                     graph_db.add_plan_instance(
#                         item_type="Member",
#                         label=m['label'],
#                         location=m['location'],
#                         associated_text=m['length_text'] or "",
#                         page_num=page_num
#                     )

#                 #2. Push the Symbol (Hexagons)
#                 for s in data["symbols"]:
#                     graph_db.add_plan_instance(
#                         item_type="Symbol",
#                         label=s["symbol"],
#                         location=s["location"],
#                         associated_text=s["associated_text"] or "",
#                         page_num=page_num
#                     )

#                 # 3. Push Schedules (If found on plan)
#                 if data["content_type"] == "Definition_Schedule":
#                     for sched in data["schedules"]:
#                         # We treat schedule rows as Rules
#                         # You might need to parse the schedule string here or just dump it
#                         graph_db.add_schedule_rule(
#                             schedule_name=sched["name"],
#                             symbol="ALL", # Placeholder, or parse specific rows
#                             specs=sched["data"],
#                             page_num=page_num
#                         )


#                 # --- DYNAMIC ROUTING LOGIC --- langgraph_temp_workflow.common.check_prev_result
                
#                 # Case A: It's a Plan View (The Map) -> Store in Raw Data
#                 if data["content_type"] == "Plan_View":
#                     # Only keep relevant fields to save space
#                     clean_data = {
#                         "sheet_type": "Plan View",
#                         "members": data["members"],
#                         "symbols": data["symbols"],
#                         "visual_reasoning": data["visual_reasoning"]
#                     }
#                     raw_plan_data.append(clean_data)
#                     print(f"  > Extracted Plan Data from crop.")

#                 # Case B: It's a Schedule (The Legend) -> Store in Detail Library
#                 elif data["content_type"] == "Definition_Schedule":
#                     for sched in data["schedules"]:
#                         # Use the Schedule Name as the Key
#                         key = sched["name"]
#                         current_detail_library[key] = {
#                             "type": "Schedule",
#                             "data": sched["data"],
#                             "source_page": page_num
#                         }
#                         print(f"  > Learned Rule: {key}")

#                 # Case C: Notes -> Store in General Rules
#                 elif data["content_type"] == "Notes":
#                     for note in data["global_notes"]:
#                         state["general_rules"] += f"\n[Page {page_num}] {note}"
#                         print(f"  > Added Note.")

#             except Exception as e:
#                 print(f"ERROR PARSING CROP: {e}")

#     # --- RETURN UPDATED STATE ---
#     return {
#         "raw_plan_data": raw_plan_data,
#         "detail_library": current_detail_library, # Updated with new schedules
#         "general_rules": state["general_rules"]   # Updated with new notes
#     }

from langgraph_temp_workflow.common.utils import crop_union_tables

# def node_process_plans(state: ProjectState):
#     print("--- NODE: Agent 2 (Plan Scanner & Ingestion) ---")
    
#     raw_plan_data = []
#     current_detail_library = state.get("detail_library", {}).copy()
#     floor_pages = [p for p, t in state["page_map"].items() if t == "floor"]
    
#     for page_num in floor_pages:
#         print(f"Processing Page {page_num}...")
        
#         # 1. Setup Paths
#         page_dir = f"{state['output_dir']}/floor_{page_num}"
#         page_img_path = f"{page_dir}.png"
#         page_pdf_path = f"{page_dir}.pdf"
        
#         # MinerU Paths (Using 'vlm' backend structure)
#         mineru_output_dir = f"{state['output_dir']}/floor_{page_num}"
#         # Note: MinerU creates a subfolder named after the PDF file
#         # If pdf is 'floor_0.pdf', subfolder is 'floor_0'
#         mineru_vlm_dir = f"{mineru_output_dir}/vlm" # Assuming you use 'vlm' backend
        
#         json_path = f"{mineru_vlm_dir}/floor_{page_num}_content_list_v2.json"
#         images_dir = f"{mineru_vlm_dir}/images"

#         # 2. Prepare Page Image & PDF
#         convert_specific_page_to_png(state["pdf_path"], page_num, page_img_path, dpi=300)
    
#         try:
#             reader = PdfReader(state["pdf_path"])
#             writer = PdfWriter()
#             writer.add_page(reader.pages[page_num])
#             with open(page_pdf_path, "wb") as f: writer.write(f)
#         except Exception as e:
#             print(f"PDF extraction failed: {e}")
#             continue

#         # 3. Run MinerU (VLM Backend)
#         print(f"   > Running MinerU (VLM) on Page {page_num}...")

#         minerU_pdf_creating_extration(page_pdf_path, state["output_dir"],"vlm-auto-engine")

#         # 4. Run Union Cropping (Title + Table Merge)
#         # This deletes old table images and creates new UNION crops in 'images_dir'
#         if os.path.exists(json_path):
#             print(f"   > Running Union Cropping...")
#             crop_union_tables(json_path, page_img_path, output_dir=images_dir)
#         else:
#             print(f"   ! MinerU JSON not found at {json_path}. Skipping Union Crop.")

#         # 5. PHASE A: INGEST SCHEDULES (From Crops)
#         # We iterate through ALL images in the folder (Union + Original)
#         if os.path.exists(images_dir):
#             image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
#             print(f"   > Found {len(image_files)} crops to analyze.")
            
#             for img_file in image_files:
#                 crop_path = os.path.join(images_dir, img_file)
                
#                 # Prompt: Extract Data
#                 prompt = """
#                 You are a Structural Data Ingestor. Analyze this image.
                
#                 ### TASK:
#                 1. **CLASSIFY:** Is this a **Schedule/Table** or **Keyed Notes**?
#                    - If it is a Floor Plan drawing -> Return "Ignore".
#                 2. **EXTRACT:** If Schedule/Notes, extract the Data Rows.
#                 3. **CRITICAL:** Look for Symbols (Hexagons, Circles) in the Header or First Column.
#                    - If found, format Key as SHAPE-VALUE (e.g. HEX-1).
                
#                 Return JSON: { "type": "Schedule" | "Ignore", "title": "...", "items": [...] }
#                 """
                
#                 msg = HumanMessage(content=[
#                     {"type": "text", "text": prompt},
#                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{load_image_base64(crop_path)}"}}
#                 ])
                
#                 try:
#                     # Use a simple schema for ingestion
#                     class IngestionOutput(BaseModel):
#                         content_type: Literal["Schedule", "Keyed_Notes", "Ignore"]
#                         title: Optional[str]
#                         items: List[Dict[str, str]] # Simple list of dicts for flexibility

#                     result = llm_flash.with_structured_output(IngestionOutput).invoke([msg])
                    
#                     if result.content_type in ["Schedule", "Keyed_Notes"]:
#                         print(f"     > Ingested: {result.title}")
#                         # Store in GraphDB
#                         for entry in result.items:
#                             # Assuming entry has 'key_id' and 'specs' keys from prompt instruction
#                             key = entry.get("key_id") or entry.get("key") or "UNKNOWN"
#                             val = entry.get("specs") or entry.get("value") or ""
                            
#                             graph_db.add_schedule_rule(
#                                 project_id=os.path.basename(state["pdf_path"]),
#                                 schedule_name=result.title,
#                                 symbol=key,
#                                 specs=val,
#                                 page_num=page_num
#                             )
#                 except Exception as e:
#                     # print(f"     ! Failed to ingest {img_file}: {e}")
#                     pass

#         # 6. PHASE B: SCAN FLOOR PLAN (From Full Image)
#         # Now we look at the big picture to find instances
#         print(f"   > Scanning Full Floor Plan...")
        
#         prompt = prompt_for_node_process_plans()
        
#         msg = HumanMessage(content=[
#             {"type": "text", "text": prompt},
#             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{load_image_base64(page_img_path)}"}}
#         ])
        
#         try:
#             result = llm_pro.with_structured_output(PlanExtraction).invoke([msg])
#             data = result.model_dump()
            
#             # Store Instances in GraphDB
#             for m in data["members"]:
#                 graph_db.add_plan_instance(
#                     os.path.basename(state["pdf_path"]),
#                     "Member", m['label'], m['location'], m['length_text'] or "", page_num
#                 )
            
#             for s in data["symbols"]:
#                 graph_db.add_plan_instance(
#                     os.path.basename(state["pdf_path"]),
#                     "Symbol", s['symbol'], s['location'], s['associated_text'] or "", page_num
#                 )
                
#             # Store in State for Agent 4
#             raw_plan_data.append(data)
            
#         except Exception as e:
#             print(f"   ! Plan Scan Failed: {e}")

#     return {
#         "raw_plan_data": raw_plan_data,
#         "detail_library": current_detail_library,
#         "general_rules": state["general_rules"]
#     }

def node_process_plans(state: ProjectState):
    print("--- NODE: Agent 2 (Plan Ingestion) ---")
    
    # New State Variable to hold the plan images for Agent 5
    floor_plan_images = [] 
    
    floor_pages = [p for p, t in state["page_map"].items() if t == "floor"]
    
    for page_num in floor_pages:
        print(f"Ingesting Page {page_num}...")
        
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
        
        try:
            reader = PdfReader(state["pdf_path"])
            writer = PdfWriter()
            writer.add_page(reader.pages[page_num])
            with open(page_pdf_path, "wb") as f: writer.write(f)
        except Exception as e:
            print(f"PDF extraction failed: {e}")
            continue

        # 3. Run MinerU (VLM Backend)
        print(f"   > Running MinerU (VLM) on Page {page_num}...")
        minerU_pdf_creating_extration(page_pdf_path, mineru_output_dir, "vlm-auto-engine")

        # 4. Run Union Cropping (Title + Table Merge)
        if os.path.exists(json_path):
            print(f"   > Running Union Cropping...")
            # This creates UNION crops in 'images_dir' and deletes old ones
            crop_union_tables(json_path, page_img_path, output_dir=images_dir)
        else:
            print(f"   ! MinerU JSON not found at {json_path}. Skipping Union Crop.")

        # 5. PROCESS CROPS (Ingest Schedules, Identify Plans)
        if os.path.exists(images_dir):
            image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
            print(f"   > Found {len(image_files)} crops to analyze.")
            
            for img_file in image_files:
                crop_path = os.path.join(images_dir, img_file)
                
                # Prompt: Classify & Extract
                prompt = """
                You are a Structural Data Ingestor. Analyze this image.
                
                ### TASK:
                1. **CLASSIFY:** What is this image?
                   - **"Plan_View"**: If it shows the building layout, walls, grids, or framing.
                     - **CRITICAL:** If you see Grid Lines with Bubbles (A, B, 1, 2) surrounding a drawing, this is a PLAN VIEW. Do NOT extract the bubbles as a schedule. Return "Plan_View".
                   - **"Schedule"**: If it is a structured Table/Grid of data rows.
                   - **"Keyed_Notes"**: If it is a list of numbered notes.
                   - **"Ignore"**: If it is a Logo, Title Block, or Noise.
                
                2. **EXTRACT (ONLY If Schedule/Notes):** Extract the Data Rows.
                   - **STOP:** If you classified it as "Plan_View" or "Ignore", return an empty `items` list `[]`. Do NOT try to read the drawing symbols here.
                
                ### CRITICAL: SYMBOL DETECTION (Visual Check)
                (Only for Schedules/Notes)
                Look at the **First Column** (or the Note Number).
                - Do NOT just read the text (e.g. "1").
                - Look at the **Shape** around the text.
                - Is it inside a **Hexagon**? -> Key = "HEX-1"
                - Is it inside a **Circle**? -> Key = "CIR-1"
                - Is it inside a **Square**? -> Key = "SQR-1"
                - Is it inside a **Triangle**? -> Key = "TRI-1"
                - If no shape, just use the text (e.g. "F5").
                
                Return JSON: 
                { 
                    "type": "Schedule" | "Keyed_Notes" | "Plan_View" | "Ignore", 
                    "title": "...", 
                    "items": [
                        {"key_id": "HEX-1", "specs": "5/8 bolt..."},
                        {"key_id": "F5", "specs": "5x5 footing..."}
                    ] 
                }
                """
                msg = HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{load_image_base64(crop_path)}"}}
                ])
                
                try:
                    # Use a simple schema for ingestion
                    class ScheduleItem(BaseModel):
                        key_id: str
                        raw_label: Optional[str] = ""
                        specs: Optional[str] = "" # Make this optional

                    class IngestionOutput(BaseModel):
                        content_type: Literal["Schedule", "Keyed_Notes", "Plan_View", "Ignore"] = Field(alias="type")
                        title: Optional[str]
                        items: List[ScheduleItem] # Use the class, not Dict
                    result = llm_flash.with_structured_output(IngestionOutput).invoke([msg])
                    
                    # CASE A: It is a Schedule/Note -> Store in Graph
                    if result.content_type in ["Schedule", "Keyed_Notes"]:
                        print(f"     > Ingested Schedule: {result.title}")
                        print(f"     > Found {len(result.items)} items.")
                        for entry_obj in result.items:
                            entry = entry_obj.model_dump()
                            key = entry.get("key_id") or entry.get("key") or "UNKNOWN"
                            val = entry.get("specs") or entry.get("value")
                            if not val:
                                # Fallback: Dump the whole dict excluding the key
                                val = str({k:v for k,v in entry.items() if k not in ["key_id", "key"]})

                            if key == "UNKNOWN":
                                print(f"     ! Warning: VLM returned UNKNOWN key for item: {entry}")

                            
                            graph_db.add_schedule_rule(
                                project_id=os.path.basename(state["pdf_path"]),
                                schedule_name=result.title,
                                symbol=key,
                                specs=val,
                                page_num=page_num
                            )

                    # CASE B: It is a Plan View -> Save for Agent 5
                    elif result.content_type == "Plan_View":
                        print(f"     > Found Floor Plan Crop: {img_file}")
                        floor_plan_images.append(crop_path)

                except Exception as e:
                    print(f"     ! Failed to ingest {img_file}: {e}")
                    

    # --- UPDATE STATE ---
    # We pass the list of identified Floor Plan images to the next node (Agent 5)
    return {
        "floor_plan_images": floor_plan_images, 
        "general_rules": "Updated Graph with Schedules"
    }

# --- 8. AGENT 4: THE LOGIC MERGER ---

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph_temp_workflow.tools.graph_tools import lookup_symbol_definition, submit_final_estimate
from langgraph_temp_workflow.common.utils import load_image_base64
# Import the new utility
from utils.symbol_detection import detect_and_read_symbols 
import pandas as pd

def get_valid_materials_list(excel_path):
    try:
        # Assuming 'Options' sheet, Column A has the list
        df = pd.read_excel(excel_path, sheet_name="Options")
        # Clean and return list
        return df.iloc[:, 0].dropna().astype(str).tolist()
    except:
        return [] # Fallback if file missing


# def node_agent_4_merger(state: ProjectState):
#     print("--- NODE: Agent 4 (Vision + ReAct Estimator) ---")
    
#     project_id = os.path.basename(state["pdf_path"])
#     excel_path="#1A Steel Estimator (2023) (1).xlsx"
#     floor_images = state.get("floor_plan_images", [])
#     valid_materials = get_valid_materials_list(excel_path)
#     valid_materials_str = json.dumps(valid_materials)
#     print(valid_materials_str)

#     if not floor_images:
#         return {"final_bill_of_materials": {"error": "No floor plans found."}}

#     # --- STEP A: PRE-PROCESS SYMBOLS (DINO + GROQ) ---
#     # We build a map of symbols to help Gemini focus
#     detected_map = {}
#     for img_path in floor_images:
#         # Save crops to a temp folder inside the floor folder
#         symbol_out_dir = os.path.join(os.path.dirname(img_path), "detected_symbols")
#         symbols = detect_and_read_symbols(img_path, symbol_out_dir)
#         detected_map[os.path.basename(img_path)] = symbols

#     # --- STEP B: SETUP REACT AGENT ---
#     tools = [lookup_symbol_definition, submit_final_estimate]
#     all_extracted_items = []
#     system_prompt = f"""
#     You are a Senior Structural Estimator.
#     Your goal is to generate a Bill of Materials (BOM) that can be directly imported into our pricing software.
    
#     ### INPUTS:
#     1. **Floor Plan Images:** Visual drawings of the structure.
#     2. **Detected Symbols List:** A pre-computed list of symbols (Hexagons, Circles) and their locations.
#     3. **Valid Material List:** A strict list of material codes allowed in our system.
    
#     ### CRITICAL CONSTRAINT: MATERIAL MAPPING
#     When you identify a steel member, you **MUST** map it to one of the values in the **Valid Material List** below.
#     - *Example:* If you see "W24x62 Beam", you must output `material_size: "W24X62"`.
#     - *Example:* If you see "HSS 5x5x5/16", you must output `material_size: "HSS5X5X5/16"`.
#     - *Failure Case:* If the item is not in the list, set `material_size: "UNKNOWN"`.
    
#     **VALID MATERIAL LIST:**
#     {valid_materials_str}
    
#     ---
#     ### YOUR WORKFLOW (Step-by-Step):
    
#     **STEP 1: VERIFY & MEASURE (Vision)**
#     - Look at the coordinates from the "Detected Symbols List".
#     - Look at the Image around that area.
#     - **Find the Dimension:** Read the text next to the symbol (e.g., "13'-10\"", "4'-0\" R.O.").
#     - **Find the Member:** Read the label pointing to the line (e.g., "W18x35").
    
#     **STEP 2: QUERY DEFINITIONS (ReAct)**
#     - Use the tool `lookup_symbol_definition` to find the meaning of symbols.
#     - *Example:* "I see Hexagon 1. Tool says it's a Shear Wall with 5/8 bolts."
#     - *Example:* "I see Detail 7/S-3.2. Tool says it's a Ladder with MC6x15.1 rails."
    
#     **STEP 3: CALCULATE QUANTITIES (Math)**
#     - **Linear Feet (LF):**
#         - **Beams:** Sum of lengths found on plan.
#         - **Columns:** Count * 18.29 ft (Global Height).
#         - **Lintels:** Window Width + 1.33 ft.
#         - **Ladders:** Count * 2 Rails * 18.29 ft.
#     - **Quantity (Count):**
#         - **Beams/Cols:** Number of pieces.
#         - **Bolts:** Calculated from spacing rules (e.g. Wall Length / Spacing).
    
#     **STEP 4: SUBMIT (Final Output)**
#     - Call `submit_final_estimate` with the list of items.
#     - Ensure every item has a `material_size` from the valid list.
#     - Ensure `logic_trace` explains your math.
#     """
#     agent_executor = create_react_agent(llm_pro, tools)    
#     # --- STEP C: BUILD PAYLOAD ---
#     content_payload = []
#     content_payload.append({"type": "text", "text": f"Project ID: {project_id}\n\n### PRE-DETECTED SYMBOLS:\n{json.dumps(detected_map, indent=2)}"})
#     messages = [
#         SystemMessage(content=system_prompt),  # <--- System Prompt goes here now
#         HumanMessage(content=content_payload)  # <--- User Input (Images)
#     ]
#     for img_path in floor_images:
#         if os.path.exists(img_path):
#             b64 = load_image_base64(img_path)
#             content_payload.append({
#                 "type": "text",
#                 "text": f"Image Filename: {os.path.basename(img_path)}"
#             })
#             content_payload.append({
#                 "type": "image_url", 
#                 "image_url": {"url": f"data:image/png;base64,{b64}"}
#             })

#     # --- STEP D: EXECUTE ---
#     try:
#         final_result = None
        
#         # Stream execution
#         for chunk in agent_executor.stream({"messages": messages}):
            
#             if "messages" in chunk:
#                 for msg in chunk["messages"]:
#                     if isinstance(msg, AIMessage) and msg.tool_calls:
#                         for tool_call in msg.tool_calls:
#                             if tool_call["name"] == "submit_final_estimate":
#                                 final_result = tool_call["args"]
#                                 print("  > Agent submitted final estimate.")

#         if final_result:
#             # Handle Pydantic/Dict return types
#             data = final_result.model_dump() if hasattr(final_result, "model_dump") else final_result
#             return {"final_bill_of_materials": data}
#         else:
#             return {"final_bill_of_materials": {"error": "Agent did not call submit tool."}}
            
#     except Exception as e:
#         print(f"ReAct Agent Failed: {e}")
#         return {"final_bill_of_materials": {"error": str(e)}}


from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph_temp_workflow.tools.graph_tools import lookup_symbol_definition, submit_final_estimate
from langgraph_temp_workflow.common.utils import load_image_base64
from utils.symbol_detection import detect_and_read_symbols 
import pandas as pd

def get_valid_materials_list(excel_path):
    try:
        df = pd.read_excel(excel_path, sheet_name="Options")
        return df.iloc[:, 0].dropna().astype(str).tolist()
    except:
        return [] 

# def node_agent_4_merger(state: ProjectState):
#     print("--- NODE: Agent 4 (Iterative Vision + ReAct Estimator) ---")
    
#     project_id = os.path.basename(state["pdf_path"])
#     floor_images = state.get("floor_plan_images", [])
    
#     # Load Excel Options
#     excel_path = "#1A Steel Estimator (2023) (1).xlsx" # Ensure this file exists!
#     valid_materials = get_valid_materials_list(excel_path)
#     valid_materials_str = json.dumps(valid_materials)

#     if not floor_images:
#         return {"final_bill_of_materials": {"error": "No floor plans found."}}

#     # 1. Setup Tools
#     tools = [lookup_symbol_definition, submit_final_estimate]
#     agent_executor = create_react_agent(llm_pro, tools)
    
#     all_extracted_items = []

#     # 2. Iterate Images
#     for img_path in floor_images:
#         if not os.path.exists(img_path): continue
        
#         filename = os.path.basename(img_path)
#         print(f"  > Processing Image: {filename}")

#         # A. Run Symbol Detection (DINO + Groq)
#         symbol_out_dir = os.path.join(os.path.dirname(img_path), "detected_symbols")
#         try:
#             symbols = detect_and_read_symbols(img_path, symbol_out_dir)
#         except Exception as e:
#             print(f"    ! Symbol detection failed: {e}")
#             symbols = []
        
#         # B. Prepare Prompt
#         system_prompt = f"""
#         You are a Senior Structural Estimator.
#         Your goal is to generate a Bill of Materials (BOM) compatible with our pricing software.
        
#         ### INPUTS:
#         1. **Current Image:** A single page from the construction set.
#         2. **Detected Symbols List:** A pre-computed list of symbols (Hexagons, Circles) and their locations.
#         3. **Valid Material List:** A strict list of material codes.
        
#         ### CRITICAL CONSTRAINT: MATERIAL MAPPING
#         When you identify a steel member, you **MUST** map it to one of the values in the **Valid Material List** below.
#         - If not in list, set `material_size: "UNKNOWN"`.
        
#         **VALID MATERIAL LIST:**
#         {valid_materials_str}
        
#         ---
#         ### YOUR WORKFLOW (Step-by-Step):
        
#         **STEP 1: VALIDATE IMAGE TYPE**
#         - Is this a **Floor Plan / Roof Plan**? If NO (Detail/Legend), call `submit_final_estimate` with EMPTY list.
        
#         **STEP 2: VERIFY & MEASURE (Vision)**
#         - Look at the "Detected Symbols List" coordinates.
#         - Look at the Image to find the **Dimension Text** next to those symbols.
        
#         **STEP 3: QUERY DEFINITIONS (ReAct)**
#         - Use `lookup_symbol_definition` to find what "Hexagon 1" or "Detail 7" means.
        
#         **STEP 4: CALCULATE QUANTITIES (Math)**
#         - **Linear Feet (LF):**
#             - **Beams:** Sum of lengths.
#             - **Columns:** Count * 18.29 ft.
#             - **Lintels:** Window Width + 1.33 ft.
#             - **Ladders:** Count * 2 Rails * 18.29 ft.
#         - **Quantity (Count):**
#             - **Bolts:** (Wall Length / Spacing) + 1.
        
#         **STEP 5: SUBMIT**
#         - Call `submit_final_estimate` with the items found on THIS page.
#         """

#         # C. Prepare Payload
#         b64 = load_image_base64(img_path)
#         content_payload = [
#             {"type": "text", "text": f"Project ID: {project_id}\n\n### DETECTED SYMBOLS:\n{json.dumps(symbols, indent=2)}"},
#             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
#         ]
        
#         messages = [
#             SystemMessage(content=system_prompt),
#             HumanMessage(content=content_payload)
#         ]

#         # D. Run ReAct Loop
#         try:
#             page_result = None
#             print(f"    > Starting ReAct Loop for {filename}...")
#             combined_prompt = f"{system_prompt}\n\n{content_payload[0]['text']}"
#             final_payload = [{"type": "text", "text": combined_prompt}]
#             final_payload.append(content_payload[1]) 
#             response = agent_executor.invoke({"messages": [HumanMessage(content=final_payload)]})
#             last_msg = response["messages"][-1]
#             for msg in response["messages"]:
#                 if isinstance(msg, AIMessage) and msg.tool_calls:
#                     for tool_call in msg.tool_calls:
#                         print(f"this su tool calll : {tool_call}")
#                         if tool_call["name"] == "submit_final_estimate":
#                             page_result = tool_call["args"]
#                             print(f"Thsi is Page result = {page_result}")
#                             print(f"    > Finished {filename}")
#                             break
            
#             # for chunk in agent_executor.stream({"messages": messages}):
#             #     if "messages" in chunk:
#             #         for msg in chunk["messages"]:
#             #             if isinstance(msg, AIMessage) and msg.tool_calls:
#             #                 if msg.content:
#             #                     print(f"      [Thought]: {msg.content[:100]}...") # Print first 100 chars
#             #                 if msg.tool_calls:
#             #                     print(f"      [Tool Call]: {msg.tool_calls[0]['name']}")
#             #                 for tool_call in msg.tool_calls:
#             #                     if tool_call["name"] == "submit_final_estimate":
#             #                         page_result = tool_call["args"]
#             #                         print(f"    > Finished {filename}")

#             # E. Aggregate Results
#             # E. Aggregate Results
#             if page_result:
#                 # 1. Unwrap the 'estimation' key if present (due to tool argument name)
#                 if "estimation" in page_result:
#                     data = page_result["estimation"]
#                 else:
#                     data = page_result
                
#                 # 2. Extract the list
#                 # Handle Pydantic model dump or raw dict
#                 if hasattr(data, "model_dump"):
#                     data = data.model_dump()
                
#                 items = data.get("final_bill_of_materials", [])
                
#                 if items:
#                     # print(f"    > Found {len(items)} items.")
#                     all_extracted_items.extend(items)
#                 else:
#                     print(f"    > Skipped (Empty result).")

#         except Exception as e:
#             print(f"    ! Error processing {filename}: {e}")

#     # 3. Final Aggregation
#     return {"final_bill_of_materials": {"final_bill_of_materials": [item.model_dump() for item in all_extracted_items]}}    

def node_agent_4_merger(state: ProjectState):
    print("--- NODE: Agent 4 (Pre-Fetch Vision Estimator) ---")
    
    project_id = os.path.basename(state["pdf_path"])
    floor_images = state.get("floor_plan_images", [])
    
    # Load Excel Options
    excel_path = "#1A Steel Estimator (2023) (1).xlsx"
    valid_materials = get_valid_materials_list(excel_path)
    valid_materials_str = json.dumps(valid_materials)

    if not floor_images:
        return {"final_bill_of_materials": {"error": "No floor plans found."}}

    all_extracted_items = []

    for img_path in floor_images:
        if not os.path.exists(img_path): continue
        
        filename = os.path.basename(img_path)
        print(f"  > Processing Image: {filename}")

        # 1. Run Symbol Detection (DINO + Groq)
        symbol_out_dir = os.path.join(os.path.dirname(img_path), "detected_symbols")
        try:
            raw_symbols = detect_and_read_symbols(img_path, symbol_out_dir)
            print(f"Here is the Raw symbol we detectd : {raw_symbols}")
        except Exception as e:
            print(f"    ! Symbol detection failed: {e}")
            raw_symbols = []

        # 2. PRE-FETCH DEFINITIONS (The Fix)
    
        enriched_symbols = []
        for sym in raw_symbols:
            query_text = f"{sym['type']} {sym['content']}" 
            
            # A. Primary Lookup (Symbol -> Detail/Rule)
            matches = graph_db.semantic_search(query_text, project_id, limit=1)
            
            definition = None
            
            if matches and matches[0]['score'] > 0.80:
                definition = matches[0] # Use the full object
                
                # B. Recursive Lookup (Detail Component -> Schedule)
                if definition.get("BOM"):
                    for item in definition["BOM"]:
                        # Check for "Schedule" or "See Plan" in rule/material
                        rule_text = item.get("rule", "") or ""
                        mat_text = item.get("material", "") or ""
                        
                        if "Schedule" in rule_text or "Schedule" in mat_text:
                            print(f"    > Resolving Reference: {mat_text}")
                            
                            # Search for the Schedule (Filter by Label="Schedule" if possible)
                            # Using the material name as the query (e.g. "LOOSE LINTEL")
                            sub_matches = graph_db.semantic_search(mat_text, project_id, limit=1)
                            
                            if sub_matches:
                                # Attach the schedule data to this specific BOM item
                                item["linked_schedule_data"] = {
                                    "id": sub_matches[0]['ID'],
                                    "specs": sub_matches[0]['Specs'] # The rules!
                                }
                                print(f"      -> Found: {sub_matches[0]['ID']}")

            # Attach the fully enriched definition to the symbol
            sym['linked_definition'] = definition
            enriched_symbols.append(sym)
        print(f"    > Enriched {len(enriched_symbols)} symbols with Graph Data.")

        # 3. ONE-SHOT PROMPT (No ReAct Loop needed anymore!)
        # We give the LLM the answer key. It just does the math.
        
        system_prompt = f"""
        You are a Senior Structural Estimator.
        
        ### INPUT DATA:
        I have detected symbols on this plan and looked up their definitions in the database.
        
        **DETECTED SYMBOLS (Enriched):**
        {json.dumps(enriched_symbols, indent=2)}
        
        **VALID MATERIALS LIST:**
        {valid_materials_str}
        
        ### YOUR TASK:
        1. **Look at the Image:** Find the **Dimension Text** next to each symbol location (bbox provided).
        2. **Calculate:** Combine the `linked_definition` + `Dimension` to get the BOM.
           - *Rule:* If definition is a Rule (e.g. "Spacing"), use Wall Length / Spacing.
           - *Rule:* If definition is a Detail (e.g. "Ladder"), use Count * Components.
        3. **Map:** Ensure material names match the Valid List.
        4. EXECUTE LOGIC (The Math)**
        - Apply the specific Linear Feet formulas:
        - Extract **W/HSS Beam:** Sum of lengths found in 'Dimension' or estimated from Grid Location.
        - **W/HSS Column:** Count * 18.29.
        - **Channel (Stair/Ladder):** Count * 18.29.
        - **Angle (Lintel):** Window Width (from Dimension) + 1.33 ft.
        - **Rod (Anchor):** 
            1. Parse 'Rule_Specs' to find spacing (e.g. "16oc" -> 16 inches).
            2. Formula: `((Wall Length / Spacing) + 1) * 1.5 ft`.

        Return the Final Bill of Materials JSON.
        """

        # 4. Call LLM (Standard Invoke)
        b64 = load_image_base64(img_path)
        msg = HumanMessage(content=[
            {"type": "text", "text": system_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        ])
        
        try:
            result = llm_pro.with_structured_output(FinalEstimation).invoke([msg])
            
            if result.final_bill_of_materials:
                print(f"    > Extracted {len(result.final_bill_of_materials)} items.")
                all_extracted_items.extend(result.final_bill_of_materials)
                
        except Exception as e:
            print(f"    ! Estimation failed for {filename}: {e}")
    return {"final_bill_of_materials": {"final_bill_of_materials": [item.model_dump() for item in all_extracted_items]}}   


# def node_agent_4_merger(state: ProjectState):
#     print("--- NODE: Agent 4 (Dictionary Lookup Estimator) ---")
    
#     floor_images = state.get("floor_plan_images", [])
#     detail_library = state.get("detail_library", {}) # Use the in-memory dictionary
    
#     # Load Excel Options
#     excel_path = "#1A Steel Estimator (2023) (1).xlsx"
#     valid_materials = get_valid_materials_list(excel_path)
#     valid_materials_str = json.dumps(valid_materials)

#     if not floor_images:
#         return {"final_bill_of_materials": {"error": "No floor plans found."}}

#     all_extracted_items = []

#     for img_path in floor_images:
#         if not os.path.exists(img_path): continue
        
#         filename = os.path.basename(img_path)
#         print(f"  > Processing Image: {filename}")

#         # 1. Run Symbol Detection (DINO + Groq)
#         symbol_out_dir = os.path.join(os.path.dirname(img_path), "detected_symbols")
#         try:
#             raw_symbols = detect_and_read_symbols(img_path, symbol_out_dir)
#         except Exception as e:
#             print(f"    ! Symbol detection failed: {e}")
#             raw_symbols = []

#         # 2. PRE-FETCH DEFINITIONS (Using Dictionary)
#         enriched_symbols = []
#         for sym in raw_symbols:
#             # sym = {'type': 'hexagon', 'content': '1', ...}
            
#             # Construct Lookup Keys
#             # Try exact match first
#             key = sym['content'] # e.g. "7/S-3.2"
            
#             # Try "Shape-Value" match (for Schedules)
#             shape_key = f"{sym['type'].upper()[:3]}-{sym['content']}" # e.g. "HEX-1"
            
#             definition = "Unknown"
            
#             # A. Check Detail Library (In-Memory)
#             if key in detail_library:
#                 definition = detail_library[key]
#             elif shape_key in detail_library:
#                 definition = detail_library[shape_key]
            
#             # B. Fallback to GraphDB (Optional, if Dictionary fails)
#             # if definition == "Unknown":
#             #     matches = graph_db.semantic_search(...)
            
#             # Attach definition
#             sym['linked_definition'] = definition
#             enriched_symbols.append(sym)

#         print(f"    > Enriched {len(enriched_symbols)} symbols with Dictionary Data.")

#         # 3. ONE-SHOT PROMPT (Same as before)
#         system_prompt = f"""
#         You are a Senior Structural Estimator.
        
#         ### INPUT DATA:
#         **DETECTED SYMBOLS (Enriched):**
#         {json.dumps(enriched_symbols, indent=2)}
        
#         **VALID MATERIALS LIST:**
#         {valid_materials_str}
        
#         ### YOUR TASK:
#         1. **Look at the Image:** Find the **Dimension Text** next to each symbol location (bbox provided).
#         2. **Calculate:** Combine the `linked_definition` + `Dimension` to get the BOM.
#         3. **Map:** Ensure material names match the Valid List.
        
#         Return the Final Bill of Materials JSON.
#         """

#         # 4. Call LLM
#         b64 = load_image_base64(img_path)
#         msg = HumanMessage(content=[
#             {"type": "text", "text": system_prompt},
#             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
#         ])
        
#         try:
#             result = llm_25_pro.with_structured_output(FinalEstimation).invoke([msg])
#             if result.final_bill_of_materials:
#                 all_extracted_items.extend(result.final_bill_of_materials)
#         except Exception as e:
#             print(f"    ! Estimation failed: {e}")

#     # Inside node_agent_4_merger return statement:
    # return {"final_bill_of_materials": {"final_bill_of_materials": [item.model_dump() for item in all_extracted_items]}}   


