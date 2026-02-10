import os
import json
from dotenv import load_dotenv
# LangChain / LangGraph Imports

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# import state from the common/state file 
from langgraph_temp_workflow.common.state import ProjectState

# import schema from the common/schema file 
from langgraph_temp_workflow.common.schemas import DrawingTypeResponse 
from langgraph_temp_workflow.common.schemas import DetailExtraction 
from langgraph_temp_workflow.common.schemas import PlanExtraction 
from langgraph_temp_workflow.common.schemas import FinalEstimation 
from langgraph_temp_workflow.common.schemas import TextRulesExtraction 

#import the schemas 
from langgraph_temp_workflow.workflows.estimation.prompt import prompt_for_node_classify_pages
from langgraph_temp_workflow.workflows.estimation.prompt import prompt_for_node_process_details
from langgraph_temp_workflow.workflows.estimation.prompt import prompt_for_node_process_plans
from langgraph_temp_workflow.workflows.estimation.prompt import prompt_for_agent_4_merger


# import cv2 model for performing sementic segmentatation
from CV.doclayout import get_coordinates_of_the_segmentation

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
    accumulated_rules = ""
    
    for page_num in text_pages:
        with pdfplumber.open(state["pdf_path"]) as pdf:
            text = pdf.pages[page_num].extract_text() or ""
        
        # Text processing doesn't strictly need Pydantic as we just want a summary string
        prompt = f"Extract structural rules (Lintel schedules, Bolt spacing) from:\n{text}"
        msg = HumanMessage(content=prompt)
        try:
            result= llm_flash.with_structured_output(TextRulesExtraction).invoke([msg])
            for rule in result.rules:
                graph_db.add_schedule_rule(
                    schedule_name=rule.schedule_name,
                    symbol=rule.symbol,
                    specs=rule.specs,
                    page_num=page_num
                )
                print(f"   > Graph: Added Rule {rule.symbol} for {rule.schedule_name}")
            state["general_rules"] += "\n".join(result.general_notes)

        except Exception as e:
            print(f"Failed to parse text rules on page {page_num}: {e}")
                
    return {"general_rules": state["general_rules"]}

# --- 6. AGENT 3: DETAIL PROCESSOR ---
def node_process_details(state: ProjectState):
    print("--- NODE: Processing Details ---")
    detail_library = state.get("detail_library", {})
    section_pages = [p for p, t in state["page_map"].items() if t == "section"]
    
    for page_num in section_pages:
        print(f"Processing Page {page_num}...")
        
        page_img_path = f"{state['output_dir']}/section_page_{page_num}.png"
        page_pdf_path = f"{state['output_dir']}/section_page_{page_num}.pdf"
        
        convert_specific_page_to_png(state["pdf_path"], page_num, page_img_path, dpi=300)
        
        # --- STEP 1: EXTRACT SHEET NUMBER (e.g., S-3.2) ---
        sheet_number = get_sheet_number(page_img_path)
        print(f"   > Identified Sheet Number: {sheet_number}")

        # --- STEP 2: EXTRACT SINGLE PAGE PDF ---
        try:
            reader = PdfReader(state["pdf_path"])
            writer = PdfWriter()
            writer.add_page(reader.pages[page_num])
            with open(page_pdf_path, "wb") as f: writer.write(f)
        except Exception as e:
            print(f"PDF extraction failed: {e}")
            continue

        # --- STEP 3: GET COORDS ---
        try:
            all_coords = find_title_coordinates_from_image_and_pdf(page_pdf_path)
            coords_dict = all_coords.get('page_1', {})
        except Exception as e:
            print(f"Coord extraction failed: {e}")
            coords_dict = {}

        if not coords_dict:
            print(f"No titles matched on page {page_num}. Skipping.")
            continue

        # --- STEP 4: CROP SECTIONS ---
        try:
            cropped_sections = crop_sections_from_page(
                coords_dict, 
                page_img_path, 
                page_pdf_path, 
                f"page_{page_num}",
                base_output_dir=state['output_dir']
            )
        except Exception as e:
            print(f"Cropping failed: {e}")
            continue
        
        # --- STEP 5: ANALYZE CROPS & BUILD KEYS ---
        for crop in cropped_sections:
            title = crop['title']
            
            # Updated Prompt to find the Detail Number (Circle/Bubble)
                    
            prompt=prompt_for_node_process_details(title) 
            img_b64 = load_image_base64(crop["image_path"])
            msg = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ])
            
            try:
                result = llm_pro.with_structured_output(DetailExtraction).invoke([msg])
                # --- CONSTRUCT THE KEY (e.g., 7/S-3.2) ---
                det_num = result.detail_number
                
                if det_num and str(det_num).lower() != "null":
                    # Clean the number (remove dots/spaces)
                    det_num = str(det_num).strip().replace(".", "")
                    key = f"{det_num}/{sheet_number}"
                else:
                    # Fallback if no number found: Use Title/Sheet
                    key = f"{title}/{sheet_number}"
                
                print(f"   > Stored Detail: {key}")
                detail_library[key] = result.model_dump()
                graph_db.add_detail_bom(
                    detail_key=key, 
                    title=result.title, 
                    materials_list=result.model_dump()["materials"], # Convert to dict list
                    page_num=page_num
                )

            except Exception as e:
                print(f"Failed to parse JSON for {title}: {e}")
        
    print('Here how the all the information about the section parts are stored :- ',detail_library)
    
    return {"detail_library": detail_library}

# --- 7. AGENT 2: PLAN ESTIMATOR (RAW DATA COLLECTOR) ---
def node_process_plans(state: ProjectState):
    print("--- NODE: Agent 2 (Plan Scanner) ---")
    raw_plan_data = []
    current_detail_library = state.get("detail_library", {}).copy()
    floor_pages = [p for p, t in state["page_map"].items() if t == "floor"]
    
    for page_num in floor_pages:
        print(f"Scanning Page Index {page_num}...")
        
        page_dir = f"{state['output_dir']}/floor_{page_num}"
        
        page_img_path = f"{state['output_dir']}/floor_{page_num}.png"
        page_pdf_path = f"{state['output_dir']}/floor_{page_num}.pdf"

        semantic_crops_dir = f"{state['output_dir']}/floor_{page_num}"

        # Here created image
        convert_specific_page_to_png(state["pdf_path"], page_num, page_img_path, dpi=300)

        # #Here created pdf
        # try:
        #     reader = PdfReader(state["pdf_path"])
        #     writer = PdfWriter()
        #     writer.add_page(reader.pages[page_num])
        #     with open(page_pdf_path, "wb") as f: writer.write(f)
        # except Exception as e:
        #     print(f"PDF extraction failed: {e}")
        #     continue
        
        # child_initial_state = {
        #     "image_path": page_img_path,
        #     "pdf_path":page_pdf_path,
        #     "detected_queue": [], 
        #     "final_crops": [], 
        #     "current_retry_count": 0,
        #     "current_region_label": None, 
        #     "current_bbox": None,
        #     "output_dir": state["output_dir"], # Pass the base output dir
        #     "extracted_data": {}
        # }
        # # call to the sementic segementation code to get the focused image 
        # semantic_segmentation_app.invoke(child_initial_state, config={"recursion_limit": 150})

        #Here we will call the ml model to do the sementic segmentation
        get_coordinates_of_the_segmentation(page_img_path,semantic_crops_dir)

        # it will stores the sementic segemntation image in the folder named as floor/roof
        sementic_croped_images = [] 
        global_img_b64 = load_image_base64(page_img_path)
        
        plan_croped_image:str
        
        if os.path.exists(semantic_crops_dir):
            for f in os.listdir(semantic_crops_dir):
                if f.endswith(".png"):
                    img_path = os.path.join(semantic_crops_dir, f)

                    success = preprocess_image_inplace(img_path)
                    if success:
                        sementic_croped_images.append(img_path)
             
            print(f" > Found {len(sementic_croped_images)-1} semantic crops (processed in-place).")
        else:
            print(f" ! Warning: No semantic crops found at {semantic_crops_dir}")
        
        # --- ADVANCED PROMPT FOR AGENT 2 ---
        prompt=prompt_for_node_process_plans()

        for q in sementic_croped_images:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{global_img_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{load_image_base64(q)}"}}
            ]
            
            msg = HumanMessage(content=content)
            
            try:
                result = llm_pro.with_structured_output(PlanExtraction).invoke([msg])
                data = result.model_dump()

                #--- PUSH TO NEO4J ----
                #1. Push Members (Beams/Cols)
                
                for m in data["members"]:
                    graph_db.add_plan_instance(
                        item_type="Member",
                        label=m['label'],
                        location=m['location'],
                        associated_text=m['length_text'] or "",
                        page_num=page_num
                    )

                #2. Push the Symbol (Hexagons)
                for s in data["symbols"]:
                    graph_db.add_plan_instance(
                        item_type="Symbol",
                        label=s["symbol"],
                        location=s["location"],
                        associated_text=s["associated_text"] or "",
                        page_num=page_num
                    )

                # 3. Push Schedules (If found on plan)
                if data["content_type"] == "Definition_Schedule":
                    for sched in data["schedules"]:
                        # We treat schedule rows as Rules
                        # You might need to parse the schedule string here or just dump it
                        graph_db.add_schedule_rule(
                            schedule_name=sched["name"],
                            symbol="ALL", # Placeholder, or parse specific rows
                            specs=sched["data"],
                            page_num=page_num
                        )


                # --- DYNAMIC ROUTING LOGIC --- langgraph_temp_workflow.common.check_prev_result
                
                # Case A: It's a Plan View (The Map) -> Store in Raw Data
                if data["content_type"] == "Plan_View":
                    # Only keep relevant fields to save space
                    clean_data = {
                        "sheet_type": "Plan View",
                        "members": data["members"],
                        "symbols": data["symbols"],
                        "visual_reasoning": data["visual_reasoning"]
                    }
                    raw_plan_data.append(clean_data)
                    print(f"  > Extracted Plan Data from crop.")

                # Case B: It's a Schedule (The Legend) -> Store in Detail Library
                elif data["content_type"] == "Definition_Schedule":
                    for sched in data["schedules"]:
                        # Use the Schedule Name as the Key
                        key = sched["name"]
                        current_detail_library[key] = {
                            "type": "Schedule",
                            "data": sched["data"],
                            "source_page": page_num
                        }
                        print(f"  > Learned Rule: {key}")

                # Case C: Notes -> Store in General Rules
                elif data["content_type"] == "Notes":
                    for note in data["global_notes"]:
                        state["general_rules"] += f"\n[Page {page_num}] {note}"
                        print(f"  > Added Note.")

            except Exception as e:
                print(f"ERROR PARSING CROP: {e}")

    # --- RETURN UPDATED STATE ---
    return {
        "raw_plan_data": raw_plan_data,
        "detail_library": current_detail_library, # Updated with new schedules
        "general_rules": state["general_rules"]   # Updated with new notes
    }

# --- 8. AGENT 4: THE LOGIC MERGER ---
def node_agent_4_merger(state: ProjectState):
    print("--- NODE: Agent 4 (The Merger) ---")
    

    # DB Setup
    graph_data = graph_db.get_full_estimation_data()

    # 1. Generate the Prompt String
    # Pass the 'state' object so the function can access raw_plan_data, etc.
    prompt_text = prompt_for_agent_4_merger(graph_data)
    
    # 2. Create Message
    msg = HumanMessage(content=prompt_text)
    
    try:
        # 3. Invoke with Schema
        result = llm_pro.with_structured_output(FinalEstimation).invoke([msg])
        
        # 4. Return Dictionary
        return {"final_bill_of_materials": result.model_dump()}
        
    except Exception as e:
        print(f"Merger Failed: {e}")
        return {"final_bill_of_materials": {"error": str(e)}}

#