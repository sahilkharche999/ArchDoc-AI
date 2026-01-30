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

#import the schemas 
from langgraph_temp_workflow.workflows.estimation.prompt import prompt_for_node_classify_pages
from langgraph_temp_workflow.workflows.estimation.prompt import prompt_for_node_process_details
from langgraph_temp_workflow.workflows.estimation.prompt import prompt_for_node_process_plans
from langgraph_temp_workflow.workflows.estimation.prompt import node_agent_4_merger


# PDF & Image Processing Imports
import pdfplumber
from pypdf import PdfReader, PdfWriter
from utils.pdf_page_to_png import convert_specific_page_to_png

# --- CUSTOM UTILS ---
from utils.croped_sections import crop_sections_from_page
from utils.sementic_segmentation import semantic_segmentation_app
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
        response = llm_flash.invoke([msg])
        accumulated_rules += f"\nPage {page_num}: {extract_text_from_response(response)}\n"
        
    return {"general_rules": accumulated_rules}

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
#            
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
                
            except Exception as e:
                print(f"Failed to parse JSON for {title}: {e}")
        
    print('Here how the all the information about the section parts are stored :- ',detail_library)
    
    return {"detail_library": detail_library}

# --- 7. AGENT 2: PLAN ESTIMATOR (RAW DATA COLLECTOR) ---
def node_process_plans(state: ProjectState):
    print("--- NODE: Agent 2 (Plan Scanner) ---")
    raw_plan_data = []
    floor_pages = [p for p, t in state["page_map"].items() if t == "floor"]
    
    for page_num in floor_pages:
        print(f"Scanning Page Index {page_num}...")
        
        page_dir = f"{state['output_dir']}/floor_{page_num}"
        page_img_path = f"{state['output_dir']}/floor_{page_num}.png"
        semantic_crops_dir = f"{state['output_dir']}/floor_{page_num}"
        convert_specific_page_to_png(state["pdf_path"], page_num, page_img_path, dpi=300)
        
        # convert_specific_page_to_png(state["pdf_path"], page_num, f"{page_dir}.png", dpi=300)
        
        # Crop in quad function call 
        # crop_image_into_quad(f"{page_dir}.png", page_dir)

 
        #calling the langraph agent to get the sementic croping of the same plan
        #it save the croped image in the folder name as file_name
        
        child_initial_state = {
            "image_path": page_img_path, 
            "detected_queue": [], 
            "final_crops": [], 
            "current_retry_count": 0,
            "current_region_label": None, 
            "current_bbox": None,
            "output_dir": state["output_dir"], # Pass the base output dir
            "extracted_data": {}
        }
        # call to the sementic segementation code to get the focused image 
        semantic_segmentation_app.invoke(child_initial_state, config={"recursion_limit": 150})

        # it will stores the sementic segemntation image in the folder named as floor/roof
        sementic_croped_images = [] 
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
        
        if os.path.exists(semantic_crops_dir):
            for f in os.listdir(semantic_crops_dir):
                if f.endswith(".png"):
                    #here we will do ml/dl operation over image to convert into grey scale or Adaptive threshold
                    sementic_croped_images.append(os.path.join(semantic_crops_dir, f))

            print(f"  > Found {len(sementic_croped_images)-1} semantic crops.")
        else:
            print(f"  ! Warning: No semantic crops found at {semantic_crops_dir}")
        
        # --- ADVANCED PROMPT FOR AGENT 2 ---
        prompt=prompt_for_node_process_plans()
        
        content = [{"type": "text", "text": prompt}]
        for q in sementic_croped_images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{load_image_base64(q)}"}} )
            
        msg = HumanMessage(content=content)
        
        try:
            result = llm_pro.with_structured_output(PlanExtraction).invoke([msg])
            raw_plan_data.append(result.model_dump())
        except Exception as e:
            print(f"ERROR PARSING JSON for Page {page_num}: {e}")
    print(f"\n\n\n")
    print(f"this how look like when we parse the fllor plana dn roof plan with given section details information in the dictionary format raw_plan_data  {raw_plan_data}")
    print(f"\n\n\n")
    return {"raw_plan_data": raw_plan_data}

# --- 8. AGENT 4: THE LOGIC MERGER ---
def node_agent_4_merger(state: ProjectState):
    print("--- NODE: Agent 4 (The Merger) ---")
    
    # Construct the Prompt    
    prompt=node_agent_4_merger(state)
    
    msg = HumanMessage(content=prompt)
    
    try:
        result = llm_pro.with_structured_output(FinalEstimation).invoke([msg])
        return {"final_bill_of_materials": result.model_dump()}
    except Exception as e:
        return {"final_bill_of_materials": {"error": str(e)}}
